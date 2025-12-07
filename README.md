# 网课录播课件提取

## 注意事项

1. 当前分支主要是给我自己用的。程序现在支持 mlx 硬件加速和并行运算。我的设备是 MacBook M3 Max。
2. '--similarity'参数越大，筛出来的片子越多，反之亦然。你可能要来回试一下。
3. 程序现在支持 DEBUG 模式。此模式下，缓存文件在本次任务结束后不再被删除，便于对长录播视频调整参数（免得每一次都要跑完整个视频）。另一种方法是使用'--start_frame'和'--end_frame'参数指定需要处理的时间范围，但是我发现不是很好用，因为老师切幻灯片的时候搞不好在哪一页突然有一个元素一闪而过。
4. 程序现在支持'--interval'参数，用于指定抽帧间隔，默认值为 1 秒。也是用于解决老师突然切下一张幻灯片，导致某一个一闪而过的元素存在于抽帧间隔中而被漏掉的问题。
5. 程序现在支持指定差异识别算法。具体见下。
6. 程序现在在待处理视频文件同路径下创建存储临时文件的路径，并且每次处理新的图像时会在该路径下创建'时间戳\_抽帧间隔\_哈希值'的子路径，并在该子路径中存放抽出来的图。程序每次运行之后会先算待处理视频文件的哈希，查找其对应的缓存路径有没有。有就直接从缓存路径中读取，没有就抽帧并比较，避免每次调参数都重新抽帧。

## 快速上手

```shell
# 1. 激活虚拟环境
source .venv/bin/activate

# 2. 运行工具（常用的需要调整的参数已经申明了）

python -m video2ppt.video2ppt --similarity 0.96 --metric ssim '/Users/pengyu/Downloads/牙体牙髓/040-口腔内科学牙体牙髓病学-第1单元-第1节.mp4' --interval 0.5 --debug
python -m video2ppt.video2ppt --similarity 0.8 --metric phash '/Users/pengyu/Downloads/牙体牙髓/040-口腔内科学牙体牙髓病学-第1单元-第1节.mp4' --debug
python -m video2ppt.video2ppt --similarity 0.7 --metric hist ''
python -m video2ppt.video2ppt --similarity 0.8 --metric ahash ''

evp --similarity 0.6 --pdfname output.pdf --start_frame 00:00:09 --end_frame 00:00:30 ./demo ./demo/demo.mp4
```

## '--metric'和'--similarity'的设置

参数是gpt-5-high自己试出来的。但是我感觉漏得太多了。记得自己调。

1. 默认：ssim 0.65
2. 更稳健但对布局变化较敏感时需更高阈值：phash 0.8
3. 保守可用，但存在过选风险：hist 0.7
4. 可增加'--min_gap 3~5'，在翻页密集时减少近邻重复、降低比较负担。比如'ssim --similarity 0.65 --min_gap 5'。
5. 更轻量：ahash 0.8

## 不同算法的用例

- aHash/pHash 每对帧计算开销极低（小尺寸、简单 DCT 或均值哈希），适合大批量比较与低间隔抽帧。
- SSIM 略重但结构敏感，已采用全局灰度简化版本，复杂度仍可接受。
- min_gap 通过跳过近期帧减少比较次数，在多页连续场景下效果明显。
