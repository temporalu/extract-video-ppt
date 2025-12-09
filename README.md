# 网课录播课件提取

## 备忘

1. 当前分支以自用为主。
2. 程序可以一次传入多个视频文件，支持 mlx 硬件加速和并行运算。记得根据设备资源调整性能限制。
2. '--similarity'相似度阈值，默认0.6。阈值越高，筛出来的幻灯片越多，反之亦然。
3. '--debug'调试模式，默认False（不传入）。调试模式下缓存文件不会被删除。
4. '--interval'抽帧间隔（秒），默认1.0。用于解决存在于抽帧间隔中的元素一闪而过而被漏掉的问题。
5. 程序支持指定相似度识别算法。具体见下。
6. 程序在待处理视频文件同路径下创建存储临时文件的路径，并且每次处理新的图像时会在该路径下创建'时间戳\_抽帧间隔\_哈希值'的子路径，并在该子路径中存放抽出来的图。程序每次运行之后会先算待处理视频文件的哈希，查找其对应的缓存路径有没有。有就直接从缓存路径中读取，没有就抽帧并比较，避免每次调参数都重新抽帧。
7. 程序支持选择是将同一组相似帧中时序最靠前的那一张还是时序最靠后的那一张作为幻灯片（即'--pick_mode'参数，不是必须传入的参数）：‘1’表示选择一组高相似帧中的最早一张，是默认值；‘2’表示选择最晚一张。最早的一张一般是干净的课件，最晚的一张一般是有老师笔记的课件。

## 快速上手

### 配置环境

```bash
# 如果没有装uv的话，先装一下

pip install uv

# 创建和初始化虚拟环境

uv venv

source .venv/bin/activate

uv pip install -e .
```

### 运行工具

```bash
# 激活虚拟环境
source .venv/bin/activate

# 运行工具（常用的需调整参数已经申明了）

python -m video2ppt.video2ppt --similarity 0.96 --metric ssim '/Users/pengyu/Downloads/修复2' --interval 0.5
python -m video2ppt.video2ppt --pick_mode 2 --similarity 0.96 --metric ssim '同上' --interval 0.5 --debug
python -m video2ppt.video2ppt --similarity 0.8 --metric phash '同上' --debug
python -m video2ppt.video2ppt --similarity 0.7 --metric hist '同上'
python -m video2ppt.video2ppt --similarity 0.8 --metric ahash '同上'

# deprecated
evp --similarity 0.6 --pdfname output.pdf --start_frame 00:00:09 --end_frame 00:00:30 ./demo ./demo/demo.mp4
```

## '--metric'相似度算法

每个老师做 ppt 讲 ppt 的风格不一样，记得自己试一下最合适的。

1. 均衡：ssim 0.65
2. 更稳健但对布局变化较敏感时需更高阈值：phash 0.8
3. 保守可用，但存在过选风险：hist 0.7
4. 可增加'--min_gap 3~5'，在翻页密集时减少近邻重复、降低比较负担。比如'ssim --similarity 0.65 --min_gap 5'。
5. 更轻量：ahash 0.8

## 不同算法的用例

gpt-5-high 写的，仅供参考。

- aHash/pHash 每对帧计算开销极低（小尺寸、简单 DCT 或均值哈希），适合大批量比较与低间隔抽帧。
- SSIM 略重但结构敏感，已采用全局灰度简化版本，复杂度仍可接受。
- min_gap 通过跳过近期帧减少比较次数，在多页连续场景下效果明显。
