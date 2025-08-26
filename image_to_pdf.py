import os
import glob
from PIL import Image
import argparse
from pathlib import Path

def get_image_files(folder_path, recursive=True):
    """
    获取文件夹中的所有图片文件，按文件名排序
    
    Args:
        folder_path (str): 文件夹路径
        recursive (bool): 是否递归遍历子文件夹
    
    Returns:
        list: 图片文件路径列表
    """
    # 支持的图片格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.gif', '*.webp']
    
    image_files = []
    
    if recursive:
        # 递归遍历所有子文件夹
        for ext in image_extensions:
            pattern = os.path.join(folder_path, '**', ext)
            image_files.extend(glob.glob(pattern, recursive=True))
    else:
        # 只遍历当前文件夹
        for ext in image_extensions:
            pattern = os.path.join(folder_path, ext)
            image_files.extend(glob.glob(pattern))
    
    # 按文件名排序
    image_files.sort()
    
    return image_files

def convert_images_to_pdf(image_files, output_pdf_path, page_size='A4'):
    """
    将图片文件列表转换为PDF
    
    Args:
        image_files (list): 图片文件路径列表
        output_pdf_path (str): 输出PDF文件路径
        page_size (str): 页面大小，如'A4', 'Letter'等
    """
    if not image_files:
        print("没有找到图片文件！")
        return
    
    # 打开第一张图片
    first_image = Image.open(image_files[0])
    
    # 转换为RGB模式（PDF不支持RGBA）
    if first_image.mode != 'RGB':
        first_image = first_image.convert('RGB')
    
    # 准备其他图片
    other_images = []
    for img_path in image_files[1:]:
        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            other_images.append(img)
            print(f"已加载: {img_path}")
        except Exception as e:
            print(f"无法加载图片 {img_path}: {e}")
    
    # 保存为PDF
    try:
        first_image.save(
            output_pdf_path,
            "PDF",
            save_all=True,
            append_images=other_images
        )
        print(f"PDF已成功保存到: {output_pdf_path}")
        print(f"总共合并了 {len(image_files)} 张图片")
    except Exception as e:
        print(f"保存PDF时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description='将文件夹中的图片合并为PDF')
    parser.add_argument('folder_path', help='包含图片的文件夹路径')
    parser.add_argument('-o', '--output', help='输出PDF文件路径（默认为output.pdf）')
    parser.add_argument('-r', '--recursive', action='store_true', 
                       help='是否递归遍历子文件夹（默认是）')
    parser.add_argument('--no-recursive', action='store_true',
                       help='不递归遍历子文件夹')
    
    args = parser.parse_args()
    
    # 检查文件夹是否存在
    if not os.path.exists(args.folder_path):
        print(f"错误: 文件夹 '{args.folder_path}' 不存在")
        return
    
    # 确定是否递归
    recursive = args.recursive or not args.no_recursive
    
    # 获取输出文件路径
    if args.output:
        output_path = args.output
    else:
        folder_name = os.path.basename(os.path.abspath(args.folder_path))
        output_path = f"{folder_name}_images.pdf"
    
    # 获取图片文件列表
    print(f"正在扫描文件夹: {args.folder_path}")
    print(f"递归模式: {'开启' if recursive else '关闭'}")
    
    image_files = get_image_files(args.folder_path, recursive)
    
    if not image_files:
        print("在指定文件夹中没有找到图片文件！")
        print("支持的格式: jpg, jpeg, png, bmp, tiff, tif, gif, webp")
        return
    
    print(f"找到 {len(image_files)} 个图片文件:")
    for i, img_path in enumerate(image_files, 1):
        print(f"  {i}. {img_path}")
    
    # 转换为PDF
    print(f"\n开始合并图片到PDF...")
    convert_images_to_pdf(image_files, output_path)

if __name__ == "__main__":
    # 如果没有命令行参数，提供交互式界面
    import sys
    if len(sys.argv) == 1:
        print("图片转PDF工具")
        print("=" * 50)
        
        folder_path = input("请输入包含图片的文件夹路径: ").strip()
        if not folder_path:
            print("未输入路径，程序退出")
            sys.exit()
        
        output_path = input("请输入输出PDF文件路径（直接回车使用默认名称）: ").strip()
        if not output_path:
            folder_name = os.path.basename(os.path.abspath(folder_path))
            output_path = f"{folder_name}_images.pdf"
        
        recursive_input = input("是否递归遍历子文件夹？(y/n，默认y): ").strip().lower()
        recursive = recursive_input != 'n'
        
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            print(f"错误: 文件夹 '{folder_path}' 不存在")
            sys.exit()
        
        # 获取图片文件列表
        print(f"正在扫描文件夹: {folder_path}")
        print(f"递归模式: {'开启' if recursive else '关闭'}")
        
        image_files = get_image_files(folder_path, recursive)
        
        if not image_files:
            print("在指定文件夹中没有找到图片文件！")
            print("支持的格式: jpg, jpeg, png, bmp, tiff, tif, gif, webp")
            sys.exit()
        
        print(f"找到 {len(image_files)} 个图片文件:")
        for i, img_path in enumerate(image_files, 1):
            print(f"  {i}. {img_path}")
        
        # 转换为PDF
        print(f"\n开始合并图片到PDF...")
        convert_images_to_pdf(image_files, output_path)
    else:
        main()
