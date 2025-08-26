# 图片转PDF工具

这是一个Python工具，用于将文件夹中的图片文件按顺序合并到一个PDF文件中。

## 功能特点

- 支持多种图片格式：JPG, JPEG, PNG, BMP, TIFF, TIF, GIF, WEBP
- 可以递归遍历子文件夹
- 按文件名自动排序
- 支持命令行和交互式两种使用方式
- 自动处理图片格式转换（确保PDF兼容性）

## 安装依赖

```bash
pip install -r requirements.txt
```

或者直接安装Pillow：

```bash
pip install Pillow
```

## 使用方法

### 方法1: 交互式使用（推荐）

直接运行脚本，按提示输入：

```bash
python image_to_pdf.py
```

程序会提示您输入：
1. 包含图片的文件夹路径
2. 输出PDF文件路径（可选）
3. 是否递归遍历子文件夹

### 方法2: 命令行使用

```bash
# 基本用法
python image_to_pdf.py "文件夹路径"

# 指定输出文件
python image_to_pdf.py "文件夹路径" -o "输出文件.pdf"

# 不递归遍历子文件夹
python image_to_pdf.py "文件夹路径" --no-recursive

# 递归遍历子文件夹（默认）
python image_to_pdf.py "文件夹路径" -r
```

### 方法3: 在代码中使用

```python
from image_to_pdf import get_image_files, convert_images_to_pdf

# 获取图片文件列表
image_files = get_image_files("文件夹路径", recursive=True)

# 转换为PDF
convert_images_to_pdf(image_files, "输出文件.pdf")
```

## 使用示例

### 示例1: 处理res文件夹中的所有图片

```bash
python image_to_pdf.py res
```

### 示例2: 处理特定子文件夹

```bash
python image_to_pdf.py "res/密度图" --no-recursive
```

### 示例3: 运行示例脚本

```bash
python example_usage.py
```

## 支持的图片格式

- JPG/JPEG
- PNG
- BMP
- TIFF/TIF
- GIF
- WEBP

## 注意事项

1. 图片会按文件名字母顺序排列
2. 所有图片都会自动转换为RGB模式以确保PDF兼容性
3. 如果图片无法加载，程序会跳过该图片并继续处理其他图片
4. 输出PDF文件会保存在当前工作目录下

## 文件说明

- `image_to_pdf.py`: 主程序文件
- `example_usage.py`: 使用示例
- `requirements.txt`: 依赖包列表
- `README.md`: 说明文档

## 故障排除

如果遇到问题，请检查：

1. 是否已安装Pillow库
2. 文件夹路径是否正确
3. 文件夹中是否包含支持的图片格式
4. 是否有足够的磁盘空间保存PDF文件
