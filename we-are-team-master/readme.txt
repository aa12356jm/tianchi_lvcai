跑densenet201修改内容：
main_nasnet.py:1，351行进行注释掉，352行解开注释；2，400行与418行，transforms.RandomCorp(331)改为224；3，batchsize根据自己服务器配置进行设置