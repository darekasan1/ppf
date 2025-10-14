# PPF - The emulator of the prophet-5
AIが作ったソフトウェアシンセサイザー。
# ビルド
基本的には[リリース](https://github.com/darekasan1/ppf/releases)のやつでいけますが、信頼できない、実行できない場合はビルドしてください。
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
make
```
# リリースにあるやつはどうなん？
リリースのやつは全てx86_64用です。また、依存をなくすためNEONやAVXへの最適化などありません。最適化したいならコードの一部を書き直してビルドしてください。
# 使ったAIは？
Claude、Gemini、ChatGPT
