#  Git常用操作教程

## 1. sshkey

​		本地 Git仓库和GitHub仓库之间的传输是通过SSH加密的，在配置SSH key之后，上传代码到Github远程仓库时就不用输入密码了。步骤如下：

  - __命令：ssh-keygen -t rsa -C "通常是你的邮箱地址"__

  - 之后会确认保存目录、提示输入密码和确认密码，不用输入直接按两下回车即可。此时sshkey就生成好了，保存在刚刚询问过的：**用户目录/.ssh/id_rsa.pub**中，使用vim或直接打开复制sshkey，一般以`ssh-rsa`开头。

    提示：mac下.ssh文件是隐藏的，可以在终端输入**open ~/.ssh**查看，也可以**cd 找到其所在目录，ls -a**查看所有可见/不可见目录。

- 登录Github–>点击头像–>Settings–>SSH and GPG keys–>选择ssh keys上的New SSH keys，填入刚刚复制的内容。

  验证：ssh -T git@github.com，若成功会提示Hi! You've successfully authenticated, ...

  