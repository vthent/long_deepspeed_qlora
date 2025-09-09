# long_deepspeed_qlora
该项目最大的作用是在普通的家用设备上能够对大模型进行深度的训练，完成之前需要在专业的H100等设备上才能进行的训练工作，适合低成本训练，如果你有8XH100，训练小模型就不要考虑了。用这个技术可以充分利用低成本设备，跨阶层进行训练。经过实际的测试，在4x3090上，对qwen3 14B模型进行微调，用2GB的文本进行训练， --lora_r 32   --lora_alpha 64  --max_seq_length 20480 可以稳定运行。


在 https://github.com/wdlctc/mini-s.git  的基础上进行修改，修改了源代码中的一些错误，兼容deepspeed和qlora，在4X3090的设备上，max_seq_length最大可以在22k上稳定运行。兼容目前为止（2025年9月9日）最新的框架。
requirements.txt 是我日常的开发环境，里面的安装包比较多。flash_attn要根据具体的环境进行安装。
技术原理：Paper: https://www.arxiv.org/abs/2407.15892

这个框架修改了很多默认的运算，需要配合以下的类工作：
EnhancedLossLoggingCallback 用来记录损失函数，
ClipAuditCallback 用来记录梯度
MST_Trainer用来配合MiniSequence进行训练。
