[toc]



### 1、配置用户名/邮箱

```bash
git config --global user.name 'jipenghui'
git config --global user.email 'XXXXXX@qq.com'
```

### 2、git init

```bash
git init .
git init demo/
```

### 3、git add

> add命令给文件添加追踪, 未被追踪的文件没有办法commit

```bash
# 在git项目里新建文件main.py
touch main.py

# 查看git状态
$ git status
> On branch master

> No commits yet

> Untracked files:
>   (use "git add <file>..." to include in what will be committed)
>         main.py

> nothing added to commit but untracked files present (use "git add" to track)

# 发现提示该文件并未被追踪
git add . 或者 git add main.py

# 查看状态
$ git status
> On branch master

> No commits yet

> Changes to be committed:
>   (use "git rm --cached <file>..." to unstage)
>         new file:   index.py
>         new file:   main.py

```

### 4、git commit

```bash
git commit -m '描述'
# 输出
> [master (root-commit) add797b] test
>  2 files changed, 0 insertions(+), 0 deletions(-)
>  create mode 100644 index.py
>  create mode 100644 main.py
```

`add797b`为**档案号**，每**commit**一次，就会生成新的**档案号**

**当文件被修改后，再次commit：**

```bash
vi index.py

# 查看状态
$ git status
> On branch master
> Changes not staged for commit:
>   (use "git add <file>..." to update what will be committed)
>   (use "git restore <file>..." to discard changes in working directory)
>         modified:   index.py

> no changes added to commit (use "git add" and/or "git commit -a")
```

**随后，添加追踪，并commit：**

```bash
git add .

$ git commit -m '更改index.py'
> [master 09e117e] 更改index.py
>  1 file changed, 1 insertion(+)
```

```bash
# 集成add和commit
git commit -am '描述'
```

### 5、git log

```bash
git log
git log --oneline # 显示一行
git log graph
git log --pretty
git log --pretty=oneline
git log --pretty=format:"%h - %an,  %ar : %s"
git log --author="author"
```

### 6、git diff

> 追踪文件修改前后的区别

```bash
# 查看没有git add后的main.py的区别
git diff main.py

# git add后会将文件状态变为staged，
#可以添加--staged来查看该状态下文件的改动
git diff --staged
```

### 7、.gitignore

```bash
# 在项目目录下创建该文件
touch .gitignore

# 如果要忽略某些文件，则可以在该文件里面填写，例如：
/model
*.text
会忽略/model目录下和以text结尾的所有文件
```

```bash
# 相当于撤销add操作
git rm -r --cached [file_name]或者.
```

### 8、git rm

```bash
git rm filename		# 删除文件
git rm -r folder	# 删除目录和目录下的所有文件
```

