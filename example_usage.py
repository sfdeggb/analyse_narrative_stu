#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片转PDF工具使用示例
"""

from image_to_pdf import get_image_files, convert_images_to_pdf
import os

def example_usage():
    """使用示例"""
    
    # 示例1: 处理res文件夹中的所有图片
    print("示例1: 处理res文件夹中的所有图片")
    print("=" * 50)
    
    res_folder = "res"
    if os.path.exists(res_folder):
        # 获取所有图片文件（递归遍历）
        image_files = get_image_files(res_folder, recursive=True)
        
        if image_files:
            print(f"找到 {len(image_files)} 个图片文件:")
            for i, img_path in enumerate(image_files, 1):
                print(f"  {i}. {img_path}")
            
            # 转换为PDF
            output_pdf = "res_images.pdf"
            convert_images_to_pdf(image_files, output_pdf)
        else:
            print("在res文件夹中没有找到图片文件")
    else:
        print("res文件夹不存在")
    
    print("\n" + "=" * 50)
    
    # 示例2: 处理特定子文件夹
    print("示例2: 处理res/密度图文件夹中的图片")
    print("=" * 50)
    
    density_folder = "res/密度图"
    if os.path.exists(density_folder):
        # 只处理当前文件夹，不递归
        image_files = get_image_files(density_folder, recursive=False)
        
        if image_files:
            print(f"找到 {len(image_files)} 个图片文件:")
            for i, img_path in enumerate(image_files, 1):
                print(f"  {i}. {img_path}")
            
            # 转换为PDF
            output_pdf = "密度图_images.pdf"
            convert_images_to_pdf(image_files, output_pdf)
        else:
            print("在密度图文件夹中没有找到图片文件")
    else:
        print("密度图文件夹不存在")

if __name__ == "__main__":
    example_usage()
