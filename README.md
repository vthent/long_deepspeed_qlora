# long_deepspeed_qlora
在 https://github.com/wdlctc/mini-s.git  的基础上进行修改，修改了源代码中的一些错误，兼容deepspeed和qlora，在4X3090的设备上，max_seq_length最大可以在22k上稳定运行。兼容目前为止（2025年9月9日）最新的框架。
技术原理：Paper: https://www.arxiv.org/abs/2407.15892
