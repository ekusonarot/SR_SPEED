# 超解像の処理速度を測る

FSRCNNを用いて動画を超解像する場合の処理時間を計測する．

メモリが不足する場合は，バッチサイズを1以上の小さい値に設定してください．

## 使い方

```
usage: main.py [-h] [-w WIDTH] [-t HEIGHT] [-c CHANNEL] [-l LENGTH] [-f FPS]
               [-d {cpu,gpu}] [-s SCALE_FACTOR] [-b BATCH_SIZE]
```

## オプション
| 省略形    | 正規形    | 意味      | デフォルト |
| ---- | ---- | ---- | ---- |
| -w | --width | 動画フレームの幅 | 144px
| -t | --height | 動画フレームの高さ | 256px
| -c | --channel | 動画のチャネル数 | 1channel
| -l | --length | 動画の時間 | 5s
| -f | --fps | 動画のフレームレート | 24fps
| -d | --device | 実行するデバイス (cpu or gpu) | cpu
| -s | --scale_factor | 超解像倍率 | 4
| -b | --batch_size | バッチサイズ | 0 (0の場合はすべてのフレームを一度に超解像する)
