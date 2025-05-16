from typing import Tuple
from dataclasses import dataclass
from model.superpoint import SuperPoint
from model.lightglue import LightGlue
from server.mavlink import CustomSITL
from utils.logger import Logger
from utils.pair import inference, get_center_aim, pixel_to_geolocation, visualize_and_save_matches, get_bbox_geo, visualize_and_save_bboxes
from torchvision.transforms import ToTensor
from utils.pair import (crop_geotiff_by_center_point, save_coordinates_to_csv, get_m_nums, crop_geotiff_by_pixel_point)
from utils.elevation import get_elevation_from_hgt, get_elevation_bilinear
import argparse
import torch
import timeit
import numpy as np
import os
import csv
from osgeo import gdal
from server.detection import YoloConfig, YoloeDetector
from geopy.distance import geodesic


@dataclass
class AppConfig:
    device: torch.device
    args: argparse.Namespace
    extractor: SuperPoint
    matcher: LightGlue
    logger: Logger
    sitl: CustomSITL
    detector_config: YoloConfig
    detector: YoloeDetector

class OptMatch:

    def __init__(self, app_config: AppConfig):
        self.config = app_config  # 添加配置存储
        self.input_ste = gdal.Open(self.config.args.image_ste_path)
        self.output_path = self.config.args.save_path
        self.fault_path = self.config.args.fault_path

    def process_image_matching(self, image_ste, real_img):
        """核心图像匹配处理流程
        Args:
            image_ste: 卫星基准图像
            real_img: 无人机实时图像
        Returns:
            matches_S_U: 匹配的特征点对
            matches_num: 匹配的特征点对数量
            m_kpts_ste: 匹配的特征点在卫星基准图像中的索引
            m_kpts_uav: 匹配的特征点在无人机实时图像中的索引
            ste_keypoints: 卫星基准图像中的特征点
            ste_scores: 卫星基准图像中特征点的置信度
            uav_keypoints: 无人机实时图像中的特征点
            uav_scores: 无人机实时图像中特征点的置信度
            matches_scores: 匹配的特征点对的置信度
        """
        start_time = timeit.default_timer()

        matches_S_U, matches_num, m_kpts_ste, m_kpts_uav, ste_keypoints, ste_scores, uav_keypoints, uav_scores, matches_scores \
            = inference(
            image_ste, real_img,
            self.config.extractor,
            self.config.matcher,
            self.config.device
        )

        elapsed_time = (timeit.default_timer() - start_time) * 1000
        self.config.logger.log(f"推理时间: {elapsed_time:.2f} 毫秒, FPS={1000 / elapsed_time:.1f}")
        return matches_S_U, matches_num, m_kpts_ste, m_kpts_uav, ste_keypoints, ste_scores, uav_keypoints, uav_scores, matches_scores

    def save_match_keypoints_geo_to_csv(self, m_kpts_ste, ste_keypoints, ste_scores, keypoint_csv_file, img_ste_geo):
        """保存匹配到的关键点的地理信息到CSV文件"""
        table_title = ["成功匹配的特征点latitude", "成功匹配的特征点longitude"]
        file_exists = os.path.exists(keypoint_csv_file)
        with open(keypoint_csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(table_title)  # 表头
            for i, ste_keypoints in enumerate(m_kpts_ste):
                lon, lat = pixel_to_geolocation(
                    ste_keypoints[0] + 0.5,
                    ste_keypoints[1] + 0.5,
                    img_ste_geo
                )
                content = [lat, lon]
                writer.writerow(content)  # 写入图像名称和对应的坐标

    def save_keypoints_geo_to_csv(self, ste_keypoints, ste_scores, keypoint_csv_file, img_ste_geo):
        """保存关键点的地理信息到CSV文件"""
        table_title = ["卫星图特征点latitude", "卫星图特征点longitude", "卫星图特征点置信度"]
        file_exists = os.path.exists(keypoint_csv_file)
        with open(keypoint_csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(table_title)  # 表头
            for i, ste_keypoints in enumerate(ste_keypoints):
                lon, lat = pixel_to_geolocation(
                    ste_keypoints[0] + 0.5,
                    ste_keypoints[1] + 0.5,
                    img_ste_geo
                )
                content = [lat, lon, ste_scores[i]]
                writer.writerow(content)  # 写入图像名称和对应的坐标
                
    def get_elevation_data(self, lat, lon):
        """获取指定经纬度位置的高程数据
        
        Args:
            lat (float): 纬度
            lon (float): 经度
            
        Returns:
            float: 高程值（单位：米），如果无法获取则返回None
        """
        # 首先尝试使用双线性插值获取更精确的高程数据
        elevation = get_elevation_bilinear(lat, lon)
        
        # 如果双线性插值失败，尝试使用普通方法获取
        if elevation is None:
            elevation = get_elevation_from_hgt(lat, lon)
            
        if elevation is not None:
            self.config.logger.log(f"获取高程数据成功：{elevation} 米")
        else:
            self.config.logger.log(f"无法获取高程数据，使用默认高度")
            
        return elevation

    def process_ste_extractor_point(self, config: AppConfig, win_size: Tuple[int, int]):
        """处理提取器点"""
        winx, winy = win_size
        #读取self.input_ste图像并从左到右使用win_size大小的窗口进行滑窗
        for x in range(0, self.input_ste.RasterXSize, winx):
            self.config.logger.log(f"正在处理第{x}/{self.input_ste.RasterXSize}行")
            for y in range(0, self.input_ste.RasterYSize, winy):
                image_ste, img_ste_geo = crop_geotiff_by_pixel_point(
                    x, y,
                    input_tif=self.input_ste,
                    crop_size_px=winx,
                    crop_size_py=winy
                )
                # 将图像转换为Tensor并移动到指定设备
                transform = ToTensor()
                image_ste = transform(image_ste).to(self.config.device)

                # 提取两张图像的特征
                feats_ste = self.config.extractor.extract(image_ste)

                ste_keypoints, ste_scores = feats_ste['keypoints'].cpu().numpy(), feats_ste['keypoint_scores'].cpu().numpy()
                keypoint_csv_file = os.path.join(self.config.args.save_path, 'keypoint_geo.csv')
                self.save_keypoints_geo_to_csv(ste_keypoints[0], ste_scores[0], keypoint_csv_file,
                                               img_ste_geo)

    def process_frame_matching(self, config: AppConfig, position_data, frame_img, win_size: Tuple[int, int], file_name):
        """处理单帧图像匹配"""
        true_lat, true_lon, true_alt, roll, pitch, heading = position_data
        winx, winy = win_size
        try:
            image_ste, img_ste_geo, _, _ = crop_geotiff_by_center_point(
                longitude=true_lon, latitude=true_lat,
                input_tif=self.input_ste,
                crop_size_px=winx,
                crop_size_py=winy
            )
        except Exception as e:
            config.logger.log(f"图像裁剪失败: {e}")
            return None, None

        # 核心匹配处理
        matches_S_U, matches_num, m_kpts_ste, m_kpts_uav, ste_keypoints, ste_scores, uav_keypoints, uav_scores, matches_scores \
            = self.process_image_matching(
            image_ste, frame_img
        )
        #检测目标
        detector = config.detector
        labels, detections = detector.detect(frame_img)
        keypoint_csv_file = os.path.join(self.config.args.save_path, 'matched_keypoint_geo.csv')
        self.save_match_keypoints_geo_to_csv(m_kpts_ste.cpu().numpy(), ste_keypoints[0], ste_scores[0], keypoint_csv_file, img_ste_geo)

        if matches_num > 8:
            # # 如果匹配点数量大于10，则选择置信度最高的前10个点
            # if len(matches_scores) > 50:
            #     # 获取置信度排序后的索引
            #     top_indices = np.argsort(matches_scores)[-50:]
            #     # 筛选出置信度最高的前10个匹配点
            #     m_kpts_ste = m_kpts_ste[top_indices]
            #     m_kpts_uav = m_kpts_uav[top_indices]

            aim = get_center_aim(winx, winy, m_kpts_ste, m_kpts_uav, matches_scores)
            #判断是否有目标
            if len(labels) == 0:
                config.logger.log(f"未检测到目标：{file_name}")
            else:
                config.logger.log(f"检测到目标：{file_name}")
                detections = get_bbox_geo(winx, winy, m_kpts_ste, m_kpts_uav, labels, detections, img_ste_geo)
                vis_path = os.path.join(self.output_path, f"{file_name}_bbox.jpg")
                visualize_and_save_bboxes(image_ste, frame_img, m_kpts_ste, m_kpts_uav, matches_S_U, vis_path, detections, labels)
            lon, lat = pixel_to_geolocation(
                aim[0] + 0.5,
                aim[1] + 0.5,
                img_ste_geo
            )
            # 获取该位置的高程数据
            elevation = self.get_elevation_data(lat, lon)
            
            current_lat, current_lon = int(lat * 1e7), int(lon * 1e7)
            
            # 将高程数据用于后续处理
            elevation_value = elevation
            
            config.logger.log(f"匹配成功：{file_name}，地表高程：{elevation_value} 米，真高：{true_alt} 米")
            # 计算两个坐标之间的距离（单位：米）
            output_geo = (lat, lon)
            true_geo = (true_lat, true_lon)
            distance = geodesic(true_geo, output_geo).meters
            distance = np.float32(distance)
            config.logger.log(
                f"文件名：{file_name} 真实坐标: {true_lat}, {true_lon}, 计算坐标: {lat}, {lon}"
                f"距离差: {distance:.2f}米")
            
            # 更新CSV内容
            content = [
                f"{file_name}", true_lat, true_lon,
                lat, lon, 
                distance,
                matches_num,
                roll, pitch, heading, true_alt, elevation_value
            ]
            title = [
                "Image Name", "true_lat", "true_lon", 
                "compute_lat", "compute_lon", 
                "distance(m)",
                "匹配点数",
                "roll", "pitch", "heading", "altitude", "elevation(m)"
            ]
            csv_file = os.path.join(self.config.args.save_path, 'image_coordinates.csv')
            save_coordinates_to_csv(csv_file, content, title)

            vis_path = os.path.join(self.output_path, f"{file_name}.jpg")
            visualize_and_save_matches(image_ste, frame_img, m_kpts_ste, m_kpts_uav, matches_S_U, vis_path)
            return current_lat, current_lon

        else:
            vis_path = os.path.join(self.fault_path, f"{file_name}.jpg")
            content = [f"{file_name}.jpg", true_lat, true_lon, matches_num]
            title = ["Image Name", "true_lat", "true_lon", "匹配点数"]
            csv_file = os.path.join(self.fault_path, 'image_coordinates.csv')  # 使用实例配置
            save_coordinates_to_csv(csv_file, content, title)
            visualize_and_save_matches(image_ste, frame_img, m_kpts_ste, m_kpts_uav, matches_S_U, vis_path)
            config.logger.log(f"匹配失败：{file_name}.jpg, 匹配点数为: {matches_num}")
            return None, None


    def process_sim_image_data(self, config: AppConfig, position_data, win_size: Tuple[int, int], csv_file: str):
        """模拟环境下图像处理流程"""
        REAL_lat, REAL_lon, COMPUTED_alt, SIM_lat, SIM_lon = position_data
        config.logger.log(f"定位数据: 真实({REAL_lat}, {REAL_lon}), 仿真({SIM_lat}, {SIM_lon}), 高度:{COMPUTED_alt}")
        winx, winy = win_size
        output_path = config.args.save_path
        fault_path = config.args.fault_path
        # 图像裁剪处理
        start_time = timeit.default_timer()
        image_ste, img_ste_geo, _, _ = crop_geotiff_by_center_point(
            longitude=SIM_lon, latitude=SIM_lat,
            input_tif=self.input_ste,
            crop_size_px=winx,
            crop_size_py=winy
        )

        real_img, real_geo, _, _ = crop_geotiff_by_center_point(
            longitude=REAL_lon, latitude=REAL_lat,
            input_tif=self.input_ste,
            crop_size_px=winx,
            crop_size_py=winy
        )

        # 核心匹配处理
        elapsed_time = (timeit.default_timer() - start_time) * 1000
        config.logger.log(f"裁剪时间: {elapsed_time:.2f} 毫秒, FPS={1000 / elapsed_time:.1f}")
        # 核心匹配处理
        matches_S_U, matches_num, m_kpts_ste, m_kpts_uav, ste_keypoints, ste_scores, uav_keypoints, uav_scores, matches_scores = self.process_image_matching(
            image_ste, real_img
        )

        if matches_num > 8:
            aim = get_center_aim(winx, winy, m_kpts_ste, m_kpts_uav, matches_scores)

            lon, lat = pixel_to_geolocation(
                aim[0] + 0.5,  
                aim[1] + 0.5,  
                img_ste_geo
            )
            
            # 获取该位置的高程数据
            elevation = self.get_elevation_data(lat, lon)

            config.sitl.update_global_position(
                current_lat = int(lat * 1e7), 
                current_lon = int(lon * 1e7),
                current_alt = COMPUTED_alt
            )
            aim_geo = (lon, lat)
            config.logger.log(
                f"真实坐标: {REAL_lat}, {REAL_lon}, 计算坐标: {aim_geo[1]}, {aim_geo[0]}, 飞控仿真坐标: {SIM_lat}, {SIM_lon}")
            coord = aim_geo
            # 将图像名称和对应的地理坐标保存到 CSV 文件
            # 获取该位置的高程数据
            elevation = self.get_elevation_data(aim_geo[1], aim_geo[0])
            elevation_value = elevation if elevation is not None else COMPUTED_alt
            
            content = [f"{REAL_lat}_{REAL_lon}.jpg", REAL_lat, REAL_lon, aim_geo[1], aim_geo[0], elevation_value]
            title = ["Image Name", "REAL_lat", "REAL_lon", "计算地理坐标LAT", "计算地理坐标LON", "elevation(m)"]
            save_coordinates_to_csv(csv_file, content, title)

            config.logger.log(f"匹配成功：{REAL_lat}, {REAL_lon}")
            vis_path = os.path.join(output_path, f"{REAL_lat}_{REAL_lon}.jpg")
            visualize_and_save_matches(image_ste, real_img, m_kpts_ste, m_kpts_uav, matches_S_U, vis_path)
        else:
            vis_path = os.path.join(fault_path, f"{REAL_lat}_{REAL_lon}.jpg")
            visualize_and_save_matches(image_ste, real_img, m_kpts_ste, m_kpts_uav, matches_S_U, fault_path)
            directions = [(0, 1000), (0, -1000), (-1000, 0), (1000, 0)]
            aims = []
            for dx, dy in directions:
                n_coord = (SIM_lon + dx * img_ste_geo[1], SIM_lat + dy * img_ste_geo[5])
   
                image_ste, img_ste_geo, _, _ = crop_geotiff_by_center_point(
                    longitude=n_coord[0], latitude=n_coord[1],
                    input_tif=self.input_ste,
                    crop_size_px=winx,
                    crop_size_py=winy)
                matches_S_U, matches_num, m_kpts_ste, m_kpts_uav, ste_keypoints, ste_scores, uav_keypoints, uav_scores, matches_scores = self.process_image_matching(
                    image_ste, real_img
                )
                aims.append((n_coord, matches_num))
            # 选取匹配数量最高的结果
            max_aim = max(aims, key=get_m_nums)
            coord, _ = max_aim
            config.logger.log(f"搜寻失败，尝试边界拓展：{SIM_lat}, {SIM_lon}.jpg")


    def test_accuracy(self, config: AppConfig, win_size: Tuple[int, int]):
        #读取output_path下的csv文件
        csv_file = os.path.join(self.config.args.save_path, 'image_coordinates.csv')
        #csv首行为属性，第二行开始为数据
        with open(csv_file, 'r') as f:
            csv_data = f.readlines()
            csv_data = [line.strip().split(',') for line in csv_data]
            csv_data = csv_data[1:]
        #循环读取csv每一行的数据，获取图像名称和对应的地理坐标
        for line in csv_data:
            image_name = line[0]
            REAL_lat = line[1]
            REAL_lon = line[2]

            winx, winy = win_size
            real_img, real_geo, _, _ = crop_geotiff_by_center_point(
                longitude=REAL_lon, latitude=REAL_lat,
                input_tif=self.input_ste,
                crop_size_px=winx,
                crop_size_py=winy
            )
            config.logger.log(f"开始处理{REAL_lat}, {REAL_lon}")

            output_path = config.args.save_path

            #读取本地存储的matches_S_U, matches_num, m_kpts_ste, m_kpts_uav
            matches_S_U = torch.load(os.path.join(output_path, f"{REAL_lat}_{REAL_lon}_matches_S_U.pth"))
            matches_num = torch.load(os.path.join(output_path, f"{REAL_lat}_{REAL_lon}_matches_num.pth"))
            m_kpts_ste = torch.load(os.path.join(output_path, f"{REAL_lat}_{REAL_lon}_m_kpts_ste.pth"))
            m_kpts_uav = torch.load(os.path.join(output_path, f"{REAL_lat}_{REAL_lon}_m_kpts_uav.pth"))

            if matches_num > config.args.num_keypoints / 15:
                aim = get_center_aim(winx, winy, m_kpts_ste, m_kpts_uav, matches_scores=None)

                lon, lat = pixel_to_geolocation(
                    aim[0] + 0.5,
                    aim[1] + 0.5,
                    real_geo,
                )

                aim_geo = (lon, lat)
                config.logger.log(
                    f"真实地理坐标: {REAL_lat}, {REAL_lon}, 计算地理坐标: {aim_geo[1]}, {aim_geo[0]}, 计算像素坐标: {aim[0]}, {aim[1]}")
                # 将图像名称和对应的地理坐标保存到 CSV 文件
                # content = [f"{REAL_lat}_{REAL_lon}.jpg", REAL_lat, REAL_lon, aim_geo[1], aim_geo[0], aim[0], aim[1]]
                # title = ["Image Name", "REAL_lat", "REAL_lon", "计算地理坐标LAT", "计算地理坐标LON", "计算像素坐标X",
                #          "计算像素坐标Y"]
                # save_coordinates_to_csv(csv_file, content, title)
                #
                # config.logger.log(f"匹配成功：{REAL_lat}, {REAL_lon}")
                # vis_path = os.path.join(output_path, f"{REAL_lat}_{REAL_lon}.jpg")
                # visualize_and_save_matches(image_ste, real_img, m_kpts_ste, m_kpts_uav, matches_S_U, vis_path)

    def run(self):
        """主控制循环"""
        win_size = (1024, 1024)
        csv_file = os.path.join(self.config.args.save_path, 'image_coordinates.csv')  # 使用实例配置

        # 由于不用舵机控制，可不使用键盘监听控制，避免误触
        # keyboard.on_press(self.config.sitl.key_press_handler)
        # keyboard.on_release(self.config.sitl.key_release_handler)

        # 控制循环，添加一个操作，执行一段时间后，空中自动关闭GPS
        inx =0
        while True:
            inx += 1
            if inx == 5:
                self.config.sitl.SIM_GPS_DISABLE(1)

            # 获取定位数据
            self.config.sitl.refresh_msg()
            position_data = self.config.sitl.get_global_position()

            # 图像处理流程
            self.process_sim_image_data(self.config, position_data, win_size, csv_file)