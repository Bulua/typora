[toc]

# 该文件包含所有关于Github操作的实例

## 1、VSCode如何上传Github

### 1.1 本地配置用户名和邮箱

```bash
git init
git config --global user.name "xxx"    //这里xxx代表你要绑定的github的用户名
git config --global user.email "xxx"   //这里xxx代表你要绑定的github的邮箱
git config --global --list            //这里查看上述的操作是否完成，即输入回车可以看到上面的用户名和邮箱
```

### 1.2 本地生成key

```bash
ssh-keygen -t rsa -C "xxx"	//这里xxx是上面的邮箱
```

然后直接一直回车，不设置密码 （也可以设置）

查看输出信息，找到`/**/.ssh/id_rsa.pub`文件的位置，复制文件中的`key`。

### 1.3 Github中配置key

打开github主页，在用户中点击Settings

![](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/%3Cimg%20src%3D%22.imgsimage-20231009161828277.png%22%20alt%3D%22image-20231009161828277%22%20style%3D%22zoom%2080%25%3B%22%20%3E.png)

找到`SSH and GPG keys`

![20231010115115](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/20231010115115.png)

新建一个`SSH key`

![20231010115131](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/20231010115131.png)

将在`id_rsa.pub`文件中复制得到的`key`复制到里面，最后点击`Add SSH key`完成。

![20231010115146](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/20231010115146.png)

### 1.4 查看是否连接

```bash
ssh -T git@github.com
```

连接成功会提示`Hi xxxx! You've successfully authenticated,....`

### 1.5 新建github仓库

在`github`主页点击左上角的`New`按钮

![20231010115256](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/20231010115256.png)

填写仓库名称，点击创建

![20231010115317](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/20231010115317.png)

### 1.6 连接仓库

```bash
git remote add origin xxx  //xxx是图中的链接地址
```

![20231010115405](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/20231010115405.png)

### 1.7 VSCode提交文件

在输入框中填写==提交信息==，点击`Commit`

![20231010115423](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/20231010115423.png)

点击`Yes`，之后稍等片刻，等待`staged`完成之后

![20231010115441](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/20231010115441.png)

点击`Sync Changes`

![20231010115500](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/20231010115500.png)

转到github上面，发现仓库已经提交完成。

![20231010115518](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/20231010115518.png)

## 2、删除git仓库中的文件或者文件夹

### 2.1 使用命令删除文件

```bash
git rm filename		# 删除文件
git rm -r folder	# 删除目录和目录下的所有文件
```

### 2.2 commit提交

```bash
git commit -m 'msg'
```

### 2.3 push推送

```bash
git push
```

## 3、VSCode图床插件PicGo的配置

### 3.1 在插件中搜索PicGo

![20231011162116](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/20231011162116.png)

### 3.2 配置PicGo

#### 3.2.1 打开settings

![20231011162133](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/20231011162133.png)

#### 3.2.2 找到Extensions中的PicGo

![20231011162301](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/20231011162301.png)

#### 3.2.3 配置信息

![20231011162451](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/20231011162451.png)

### 3.3 仓库创建就不用说了，这里仅展示如何生成token信息

打开github，点击用户头像，找到setting

![20231011162620](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/20231011162620.png)

找到Development settings

![20231011162706](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/20231011162706.png)

选择：

![20231011162815](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/20231011162815.png)


![20231011162856](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/20231011162856.png)

![20231011162944](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/20231011162944.png)

翻到最下面点击Generate token

![20231011163024](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/20231011163024.png)

最后复制token到VSCode中PicGo配置中

### 3.4 文件上传

随机复制一张图片, 之后按==Ctrl+Alt+U==即可自动上传，上传成功后会自动出现图片的md格式。
![20231012112315](https://raw.githubusercontent.com/Bulua/BlogImageBed/master/20231012112315.png)

