# tt-ins-yt-media-cv-qc

TikTok 同款发现、博主封面采集、图片初筛及本地视频技术规格/人脸尺寸检查工具集。当前不包含 Instagram 或 YouTube 专用采集器。

**详细模块结构、阈值、技术难题、数据衔接、已知缺陷与效果边界，请阅读[流程说明](流程说明.md)。** 视频专项操作见[VIDEO_QC_README](VIDEO_QC_README.md)。

## 安装

```powershell
python -m pip install -r requirements.txt
python -m playwright install chromium
```

另需 Google Chrome、FFmpeg/FFprobe（加入PATH）。模型首次运行需要下载，浏览器登录态与权重不包含在仓库中。网络代理默认127.0.0.1:7890，请核对本机配置。详见流程说明的环境章节。

## 快速运行

在仓库目录打开PowerShell，启动独立Chrome并完成TikTok登录：

```powershell
& "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="$PWD\.tiktok_image_profile"
python .\tiktok博主图片采集.py profiles.csv --cdp --covers --output output_covers_new
```

仅检查本地视频：

```powershell
python .\check_media_specs.py "D:\视频目录"
python .\face_size_check.py "D:\视频目录"
```

单独检测图片叠字：

```powershell
python .\tiktok封面水印筛选.py "D:\待筛图片" --output "D:\筛选结果" --dry-run
```

## 文件入口

| 文件 | 功能 |
|---|---|
| tiktok拍同款抓取.py | 音乐/原视频发现博主，API优先、浏览器兜底 |
| tiktok博主图片采集.py | 博主封面下载、InsightFace/OCR/VLM初筛、Excel汇总 |
| tiktok封面水印筛选.py | OCR文字与区域启发式；也可独立处理图片 |
| vlm_overlay.py | 延迟加载SmolVLM，判断疑似编辑叠加 |
| check_media_specs.py | 视频格式、编码、尺寸、帧率、时长及图片规格 |
| face_size_check.py | YuNet检查视频10%/50%/90%三帧中的人脸尺寸 |
| test_video_qc.py | 9项视频质检回归测试 |
| profiles.csv | 原始输入，url列 |
| output_covers/ | 原有最后一轮结果：65张图片、summary.xlsx、passed_videos.csv |

## 使用边界

- A/clean仅代表机器未触发对应规则，不等于确认无水印或全片合格。
- 图文浏览器分支目前有未定义变量问题；本版推荐上述`--cdp --covers`封面路径。
- 拍同款CSV需把“博主主页”列另存为`url`列，再交给图片采集器。
- A级清单需把`video_url`列另存为“视频链接”列，再交给视频人脸脚本，避免把主页也当视频下载。
- 原采集模块的模型失败、断点等限制详见流程说明。修复后视频脚本会逐条保存报告，但无自动续跑。
- output_covers为保留快照，未重新评测。历史表内绝对路径可能需按当前目录定位。

## 验证

```powershell
python -m unittest test_video_qc -v
python .\tiktok拍同款抓取.py --self-test
```

[本次验证记录](docs/验证记录.md) · [原始结果校验清单](docs/结果文件清单.json)
