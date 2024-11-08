[toc]

# yd-ui-admin-vue3

## 1、新增菜单

找到**菜单管理**，新增菜单，如下：

<img src=".\assets\image-20241030092440976.png" alt="image-20241030092440976" style="zoom: 50%;" />

**路由地址**的意思是浏览器访问服务地址`localhost`后紧跟的字符串，如上图路由地址是`chaitin`，那么访问该菜单服务的就是`localhost/父级菜单地址/父级菜单地址/chaitin`。

**组件地址**的意思是该页面在源代码中的路径，如下图所示：

<img src=".\assets\image-20241030092907591.png" alt="image-20241030092907591" style="zoom:67%;" />

**组件名称**应与`index.vue`中的名称一致，不一致会出现该页面不会进行缓存的情况（即使不关闭该页面也会重新请求页面数据），如下图所示：

```js
defineOptions({ name: 'ChaitinReport' })
```

## 2、ChaitinReport页面开发

### 2.1 定义Api请求

在`api/ai/report`新建`index.ts`，填写以下代码：

```typescript
import request from '@/config/axios'

export const ReportApi = {

    // 测试使用
    chaitinReportTest: async () => {
      return await request.get({ url: `/ai/report/chaitin-test`})
    },

    chaitinReport: async () => {
      return await request.get({ url: `/ai/report/chaitin`})
    }
}
```

### 2.2 请求数据

在`views/ai/report/chaitin/index.vue`中，整体代码骨架如下：

```js
<template>
	前端代码 ...
</template>

<script lang="ts" setup>
import { ReportApi } from '@/api/ai/report'
    
defineOptions({ name: 'ChaitinReport' })

const getAllApi = async () => {
    ...
}

onMounted(async () => {
  getAllApi()
})
</script>
```

在`onMounted`中执行页面的所有请求，所有请求均在`getAllApi`中实现。

### 2.3 响应式变量

`loading`默认值为`true`，当发生变化时，`vue`会实时检测到`loading`的变化。

```js
const loading = ref(true)
```

### 2.4 Echarts表格生成

#### 2.4.1 前端代码

```vue
<el-skeleton :loading="loading" animated>
    <Echart class="echart" :options="pieOptionsData" :height="280" ref="pieChart" />
</el-skeleton>
```

`pieOptionsData`是该图的数据，`animated`可实现图的动画。

#### 2.4.2 Echarts各类型图的配置

为了代码可读性更高，可以将`Echarts`的各类型图的所有配置写在一个文件中，参考以下配置：

```typescript
import { EChartsOption } from 'echarts'

const { t } = useI18n()

// 默认字体大小
const defaultFontSize = 14
// 默认图表完成的动画时间
const defaultAnimationDuration = 1000

export const lineOptions: EChartsOption = {
  title: {
    text: t('analysis.monthlySales'),
    left: 'center'
  },
  xAxis: {
    data: [
      t('analysis.january'),
      t('analysis.february'),
      t('analysis.march'),
      t('analysis.april'),
      t('analysis.may'),
      t('analysis.june'),
      t('analysis.july'),
      t('analysis.august'),
      t('analysis.september'),
      t('analysis.october'),
      t('analysis.november'),
      t('analysis.december')
    ],
    boundaryGap: false,
    axisTick: {
      show: false
    },
    axisLabel: {
      fontSize: defaultFontSize,
      rotate: -45,
    },
  },
  grid: {
    left: 30,
    right: 60,
    bottom: 20,
    top: 80,
    containLabel: true
  },
  tooltip: {
    trigger: 'axis',
    axisPointer: {
      type: 'cross'
    },
    padding: [5, 10]
  },
  yAxis: {
    axisTick: {
      show: false
    },
    axisLabel: {
      fontSize: defaultFontSize, // 调整X轴字体大小
    },
  },
  legend: {
    data: [t('analysis.estimate'), t('analysis.actual')],
    top: 50
  },
  series: [
    {
      name: t('analysis.estimate'),
      smooth: true,
      type: 'line',
      data: [100, 120, 161, 134, 105, 160, 165, 114, 163, 185, 118, 123],
      animationDuration: defaultAnimationDuration,
      animationEasing: 'cubicInOut',
    }
  ]
}

export const pieOptions: EChartsOption = {
  title: {
    text: t('analysis.userAccessSource'),
    left: 'center'
  },
  tooltip: {
    trigger: 'item',
    formatter: '{a} <br/>{b} : {c} ({d}%)'
  },
  legend: {
    orient: 'vertical',
    left: 'left',
    data: [
      t('analysis.directAccess'),
      t('analysis.mailMarketing'),
      t('analysis.allianceAdvertising'),
      t('analysis.videoAdvertising'),
      t('analysis.searchEngines')
    ]
  },
  series: [
    {
      name: t('analysis.userAccessSource'),
      type: 'pie',
      radius: '70%',
      center: ['50%', '60%'],
      data: [
        { value: 335, name: t('analysis.directAccess') },
        { value: 310, name: t('analysis.mailMarketing') },
        { value: 234, name: t('analysis.allianceAdvertising') },
        { value: 135, name: t('analysis.videoAdvertising') },
        { value: 1548, name: t('analysis.searchEngines') }
      ],
      label: {
        fontSize: 16,
      }
    }
  ]
}

export const barOptions: EChartsOption = {
  title: {
    text: t('analysis.weeklyUserActivity'),
    left: 'center'
  },
  tooltip: {
    trigger: 'axis',
    axisPointer: {
      type: 'shadow'
    }
  },
  grid: {
    left: 50,
    right: 40,
    bottom: 20
  },
  xAxis: {
    type: 'value',
    axisLabel: {
      fontSize: defaultFontSize, // 调整X轴字体大小
    },
  },
  yAxis: {
    type: 'category',
    data: [
      t('analysis.monday'),
      t('analysis.tuesday'),
      t('analysis.wednesday'),
      t('analysis.thursday'),
      t('analysis.friday'),
      t('analysis.saturday'),
      t('analysis.sunday')
    ],
    axisTick: {
      alignWithLabel: true
    },
    axisLabel: {
      fontSize: defaultFontSize, // 调整X轴字体大小
    },
  },
  series: [
    {
      name: t('analysis.activeQuantity'),
      data: [13253, 34235, 26321, 12340, 24643, 1322, 1324],
      type: 'bar',
      label: {
        show: true,
        position: 'right',
        fontSize: 16,
        color: 'rgb(116,116,116)'
      }
    }
  ]
}
```

在使用到某个图的配置时，只需要引入即可，如下：

```js
import { EChartsOption } from 'echarts'
import { pieOptions, barOptions, lineOptions } from "@/views/Home/echarts-data"
```

`echarts-data`就是我们定义所有配置图表的文件，在定义数据时，可以用以下方法：

```js
const lineOptionsData = reactive<EChartsOption>(lineOptions) as EChartsOption
const pieOptionsData = reactive<EChartsOption>(pieOptions) as EChartsOption
const barOptionsData = reactive<EChartsOption>(barOptions) as EChartsOption
```

#### 2.4.3 渲染数据

使用`set`方法设置上面我们定义好变量的值，`lineOptionsData`的属性可以在`echarts-data`配置文件中查看。以下是`lineOptionsData`的例子：

```js
import { set } from "lodash-es";

const getDateCountData = async (dc) => {
  	let dateCount = dc.map(item => (
		{ value: item.count,  name: formatDateTime(item.time)}
	))
    set(lineOptionsData, 'title.text', '近一周内的攻击日志')
    set(lineOptionsData, 'legend.data', dateCount.map((v) => v.name))
    set(lineOptionsData, 'xAxis.data', dateCount.map((v) => v.name))
    set(lineOptionsData, 'series.name', '当天日期日志数量')
    set(lineOptionsData, 'series[0].data', dateCount.map((v) => v.value))
}
```

### 2.5 markdown to html

引入`markdown-it`库：

```js
import MarkdownIt from 'markdown-it'

const reportData = ref({});
const md = new MarkdownIt()

const reportData.value.htmlData = md.render(markdownData)
```

`markdownData`是`markdown`数据。前端代码

```vue
<div class="ai-content" v-html="reportData.htmlData"></div>
```

### 2.6 Chaitin报告的word下载（直接填充内容）

#### 2.6.1 引入所需要的包

```js
// 用于 html 与 docx 之间的转换
import { asBlob } from 'html-docx-js-typescript';
// 用于将图表转为 html元素
import html2canvas from "html2canvas";
// 用于保存到 docx
import { saveAs } from 'file-saver';
```

#### 2.6.2 获取页面元素

**1、通过classname获取**

```js
const echarts: Element[] = Array.from(document.getElementsByClassName('echart'));
const aiHTMLContents: Element[] = Array.from(document.getElementsByClassName('ai-content'));
```

**2、通过ref属性获取**

```js
<div ref='ele'></div>
const ele = ref()
```

第二种方式虽然更好，但是不清楚为什么获取的总是`undefined`值，可能是因为页面还没加载完的原因，但毕竟不是专业的前端，所以还是先不采用这种方式。

#### 2.6.3 图表转换

```js
const base64Images: Record<string, any>[] = [];
const aiContents: string[] = aiHTMLContents.map((item) => item.innerHTML);

const promises = echarts.map((chart: HTMLElement) =>
	html2canvas(chart).then((canvas: HTMLCanvasElement) => {
		base64Images.push({"img": canvas.toDataURL('image/png'), "h": canvas.height, "w": canvas.width});
	})
);
```

#### 2.6.4 生成word

```js
let htmlString = `<!DOCTYPE html>
            <html lang="en">
            <head>
              <meta charset="UTF-8">
              <title>Document</title>
            </head>
            <body>`
Promise.all(promises).then(() => {
    for (let i = 0; i < aiContents.length; i ++) {
        htmlString += base64ToElementImg(base64Images[i])
        htmlString += aiContents[i]
    }
    htmlString += `</body>
            </html>`
    asBlob(htmlString).then(data => {
      saveAs(data, 'report.docx')
    })
})
```

将图表和文字内容用字符串拼接的方式插入到`html`中，之后将`html`转为`word`的格式并保存到`report.docx`文件中。

下面是通过传入`base64`字符串，生成`img dom`元素的代码。

```js
const base64ToElementImg = (base64) => {
  return `<img src="${base64['img']}" alt="Base64 Image" width="${base64['w'] / 2.5}" height="${base64['h'] / 2.5}" />`
}
```

### 2.7 Chaitin报告的word下载（模板生成）

#### 2.7.1 引入所需要的包

```js
import { asBlob } from 'html-docx-js-typescript';
import html2canvas from "html2canvas";
import { saveAs } from 'file-saver';
import {
  patchDocument,
  Document,
  Packer,
  Paragraph,
  TextRun,
  ImageRun,
  PatchType,
  AlignmentType,
  HeadingLevel
} from "docx";
```

#### 2.7.2 配置模板docx

将预先定义的`template.docx`放在`src/views/ai/report/chaitin`目录中，并配置对应的占位符`{{text}}、{{img}}`。

#### 2.7.3 获取元素数据

```js
const echarts: Element[] = Array.from(document.getElementsByClassName('echart'));
const aiHTMLContents: Element[] = Array.from(document.getElementsByClassName('ai-content'));
const base64Images: Record<string, any>[] = [];
const aiContents: string[] = aiHTMLContents.map((item) => item.textContent ? item.textContent : '');
const promises = echarts.map((chart: HTMLElement) =>
	html2canvas(chart).then((canvas: HTMLCanvasElement) => {
		base64Images.push({"img": canvas.toDataURL('image/png'), "h": canvas.height, "w": canvas.width});
    })
);
```

#### 2.7.4 请求模板

因存在`js`无法直接访问本地文件，采用请求的方式来获取模板：

```js
fetch('/src/views/ai/report/chaitin/template.docx')
	.then(response => {
  		if (!response.ok) {
    		throw new Error('Network response was not ok');
  		}
  		return response.blob();
	})
    .then(blob => {
      	Promise.all(promises).then(() => {
        	fillChartsAndContents(blob, base64Images, aiContents)
      	})
    })
```

`fillChartsAndContents`方法将`base64Image`和`content`填充到`blob`模板对象中。

#### 2.7.5 数据填充

使用`patchDocument`对象来进行模板填充，固定格式样例如下：

```js
patchDocument({
    outputType: "blob",
    data: blob,		// 模板对象
    patches: {
        // 占位符名称应与模板中占位符名称相同
        img: {
            type: PatchType.PARAGRAPH,
            children: [
                new ImageRun(...),
            ]
        },
        text: {
            type: PatchType.DOCUMENT,
            children: [
                new TextRun(...),
            ]
        }
    }
})
```

下面是实现方式，插入了`3`张图片和`3`段文本内容。

```js
const base64ToImageRun = (base64: string, w: number, h: number) => {
    // 将base64字符串转为ImageRun对象
    const base64Code = base64.split(',')[1]
    const binaryString = atob(base64Code);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
		bytes[i] = binaryString.charCodeAt(i);
	}
    return new ImageRun({
        type: 'png',
        data: bytes,
        transformation: { width: w, height: h, },
    })
}

const aiContentToTextRun = (aiContent: string) => {
    // 配置文本的样式
    return new TextRun({
        text: aiContent,
        size: 24,
        bold: false,
        font: '仿宋',
        color: '000000',
    })
}

const contentSplitToParagraph = (content) => {
    // 将大段文本根据换行符进行切割，每段则是一个段落
    const aiTexts = content ? content.split('\n') : ['']
    const paragraph: Paragraph[] = []

    for (let j = 0; j < aiTexts.length; j++) {
        // 防止空文本
        if (!aiTexts[j] || aiTexts[j].length <= 0) {
			continue
        }
    	paragraph.push(new Paragraph({
            indent: { firstLine: 480 },
            spacing: { before: 0, after: 0 , line: 360},
            children: [
            	aiContentToTextRun(aiTexts[j])
            ]
    	}))
    }
    return paragraph
}

const fillChartsAndContents = (blob, echarts, contents) => {
    // 将原始图片按照一定的比例进行缩放，以便图片正常的在word中显示
    const imgScaleRatio = 2.7
    const patches = {}

    for (let i = 0; i < contents.length; i++) {
        const chart = {}
        const content = {}

        chart['type'] = PatchType.PARAGRAPH
        chart['children'] = [
            base64ToImageRun(
                echarts[i]['img'], 
                echarts[i]['w'] / imgScaleRatio, 
                echarts[i]['h'] / imgScaleRatio
            )
        ]
        content['type'] = PatchType.DOCUMENT
        content['children'] = contentSplitToParagraph(contents[i])

        patches['chart'+ (i+1)] = chart
        patches['content'+ (i+1)] = content
    }

    patchDocument({
        outputType: "blob",
        data: blob,
        patches: patches,
        // 保留原有的样式
        keepOriginalStyles: true,
    }).then((doc) => {
    	saveAs(doc, "report.docx")
    })
}
```

## 3、md转为html

需使用`markdown-it`库。

```js
import MarkdownIt from 'markdown-it'

const md = new MarkdownIt({ breaks: true })

const mdString = ''
htmlElements = md.render(mdString)
```

## 4、docx库的使用方法(js生成word)

### 4.1 patchDocument

`patchDocument`可以扫描word文档中的`{{}}`将内容插入到`{{}}`中。

`chart1、content1`分别与`{{chart1}}、{{content1}}`进行匹配。

```js
patchDocument({
    outputType: "blob",
    data: blob,
    patches: {
      chart1: {
        type: PatchType.PARAGRAPH,
        children: [
          new ImageRun({
            type: 'png',
            data: img,
            transformation: { width: w, height: h, },
          })
        ],
      },
      content1: {
        type: PatchType.DOCUMENT,
        children: [
            new Paragraph({
              indent: { firstLine: 480 },
              spacing: { before: 0, after: 0 , line: 360},
              children: [
				new TextRun({
                    text: aiContent,
                    size: 24,
                    bold: false,
                    font: '仿宋',
                    color: '000000',
              	})
              ]
            })
        ]
      },
    },
    keepOriginalStyles: true,
  }).then((doc) => {
      saveAs(doc, "report.docx")
  })
```

### 4.2 TextRun对象的定义

`TextRun`对象可进行如下定义：

```js
new TextRun({
    text: aiContent,
    size: 24,
    bold: false,
    font: '仿宋',
    color: '000000',
    highlight: 'yellow',
    strike: true,
    superScript: true,
    subScript: true,
    allCaps: true,
    smallCaps: true,
    break: 1,
})
```

- `text`指定内容。
- `size`指定内容文字大小。
- `bold`指定内容文字样式是否加粗。
- `font`指定内容文字字体。
- `color`指定内容文字颜色。
- `highlight`指定内容文字高亮颜色。
- `strike`指定内容文字是否有删除线。
- `superScript、subScript`指定上下标。
- `allCaps、smallCaps`指定内容全部大、小写。
- `break`指定将文本放在另一行文本下方但位于同一段落内。

### 4.3 ImageRun对象的定义

```js
new ImageRun({
    type: 'gif' | 'jpg' | 'bmp' | 'png' | 'svg',
    data: img,
    transformation: {
        width: 200,
        height: 200,
    },
    altText: {
        title: "This is an ultimate title",
        description: "This is an ultimate image",
        name: "My Ultimate Image",
    }
    floating: {
        horizontalPosition: {
            offset: 2014400,
        },
        verticalPosition: {
            offset: 2014400,
        },
        margins: {
            top: 201440,
            bottom: 201440,
        },
    },
})
```

- `transformation`可指定图像宽、高。
- `altText`可指定图片的名称、标题和说明。
- `horizontalPosition、verticalPosition`指定图片的水平和垂直位置。
- `margins`指定图片的外边距。

### 4.4 Paragraph对象的定义

`Paragraph`可以指定段落的样式：

```js
new Paragraph({
    indent: { 
        firstLine: 480 
    },
	spacing: { 
        before: 0, 
        after: 0 , 
        line: 360
    },	
    frame: {
        position: {
            x: 1000,
            y: 3000,
        },
        width: 4000,
        height: 1000,
        anchor: {
            horizontal: FrameAnchorType.MARGIN,
            vertical: FrameAnchorType.MARGIN,
        },
        alignment: {
            x: HorizontalPositionAlign.CENTER,
            y: VerticalPositionAlign.TOP,
        },
    },
    border: {
        top: {
            color: "auto",
            space: 1,
            value: "single",
            size: 6,
        },
        bottom: {
            color: "auto",
            space: 1,
            value: "single",
            size: 6,
        },
        left: {
            color: "auto",
            space: 1,
            value: "single",
            size: 6,
        },
        right: {
            color: "auto",
            space: 1,
            value: "single",
            size: 6,
        },
    },
    children: [
        new TextRun("Hello World"),
    ],
});

```

- `indent`: 设置段落的缩进属性，其中 `firstLine: 480` 指定了段落的首行缩进。

- `spacing`: 设置段落的行间距和段前、段后距离。

- `before` 和 `after` 控制段前和段后空白。
- `line` 指定行距，数值可能与其他属性组合使用以确保合适的布局。

- `frame`: 设置段落的框架信息。

- `position` 指定了框架的位置坐标（`x` 和 `y`）。
- `width` 和 `height` 设置框架的宽和高。
- `anchor` 用于指定锚点，控制段落相对于页面边距的位置。
- `alignment` 控制段落在页面上的水平和垂直对齐方式。

- `border`: 设置段落的边框。

- `top`, `bottom`, `left`, `right` 是各边的样式，可以设置边框颜色、大小、类型等属性。

- `children`: 段落中的内容。在这里，`new TextRun("Hello World")` 是段落的文本内容。

# yd后端

## 1、异步AI请求

当一个业务需要多次请求不同的`ai`模型时，如果是同步则比较耗费时间，实行异步则可以大大降低获取`ai`分析结果的时间。

### 1.1 配置类

新建配置类`AsyncConfig`，添加注解`@Configuration`和`@EnableAsync`开启异步功能。

```java
@Configuration
@EnableAsync
public class AsyncConfig implements AsyncConfigurer {

}
```

### 1.2 Service层的改动

需要注意的是：调用`ai`请求的`service`（调用者）和`ai`请求的`service`（被调用者）应该属于不同的`class`。

比如该业务的具体实现是`ReportService`（调用者），需要新建一个类`AiService`来实现具体的异步方法`aiAnalyze`，如下：

```java
@Service
@Slf4j
public class ReportServiceImpl implements ReportService {
    @Resource
    private AIService aiService;
    
    @Override
    public void report(String requestContent) {
        Future<String> response1 = aiService.aiAnalyze(requestContent);
        Future<String> response2 = aiService.aiAnalyze(requestContent);	// 这行的执行不会等待上面的执行结果
        Future<String> response3 = aiService.aiAnalyze(requestContent);	// 这行的执行不会等待上面的执行结果
        
        // get()方法是同步的，不要这么做 aiService.aiAnalyze(requestContent).get()
        System.out.println(response1.get());
        System.out.println(response2.get());
        System.out.println(response3.get());
    }
}

@Service
@Slf4j
public class AiServiceImpl implements AiService {

    @Override
    @Async
    public Future<String> report(String requestContent) {
        // do Something ...
    }
}
```

## 2、SQL记录

将`timestamp`时间戳转为指定格式的`SQL`语句如下：

```mysql
DATE_FORMAT(FROM_UNIXTIME(timestamp), '%Y-%m-%d %H:00:00')
```

根据`timestamp`指定近`7`天内数据的查询条件如下：

```mysql
WHERE
	stamp >= UNIX_TIMESTAMP(NOW() - INTERVAL 7 DAY)
```

判断某个字段是否为空或者`null`值的`SQL`语句如下：

```mysql
IFNULL(NULLIF(TRIM(字段名), ''), 'other')
```

*TRIM消除多余字段左右的空格，NULLIF判断如果该字段为空，则返回NULL值，如果字段为NULL值，IFNULL则将其默认设置为other值。*

将`x`保留`2`位小数的`SQL`语句：

```mysql
ROUND(x, 2)
```

## 3、对dify的请求

### 3.1 blocking 请求方式

```java
import org.json.JSONObject;
import org.springframework.http.HttpHeaders;
import org.springframework.web.reactive.function.client.WebClient;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Future;


public Future<String> aiAnalyze(String query, String apiKey) {
    String ana = "";
    WebClient webClient = WebClient.create(HOST_NAME);

    JSONObject jsonRequest = new JSONObject();
    jsonRequest.put("inputs", new JSONObject().put("query", query));  // 替换为实际的输入
    jsonRequest.put("response_mode", "blocking");
    jsonRequest.put("user", "test_user");  // 替换为实际的用户

    String responseData  = webClient
            .post()
            .headers(headers -> {
                headers.set(HttpHeaders.AUTHORIZATION, "Bearer " + apiKey);  // 设置 Authorization 头
                headers.set(HttpHeaders.CONTENT_TYPE, "application/json");
            })
            .bodyValue(jsonRequest.toString())
            .retrieve()
            .bodyToMono(String.class).block();
    if (responseData != null) {
        JSONObject jsonObject = new JSONObject(responseData);
        ana = (String) jsonObject.getJSONObject("data").getJSONObject("outputs").get("text");
    }
    return CompletableFuture.completedFuture(ana);
}
```

### 3.2 streaming 请求方式

## 4、ApiKey统一管理

在项目中的`application.yml`中设置`api-key`：

```yaml
dify:
  api-key:
    chaitin:
      date-count: app-OY2tBO7BAk7RtmM2Ij0nPEnF
      province-count: app-TXjCMb2pC6GQdAHsVqnxx0IU
      attack-type-count: app-MiZnAyT7VnsWc7ZBK6xreZd2
```

创建配置类`ChaitinApiKeyProperties`，添加注解`@ConfigurationProperties、@Component、@Data`。

- `@ConfigurationProperties`：需要指定`prefix`，代表了在`application.yml`中属性值的前缀。
- `@Component`：交给`spring`统一管理。
- `@Data`：`lombok`注解。

```java
@ConfigurationProperties(prefix = "dify.api-key.chaitin")
@Data
@Component
public class ChaitinApiKeyProperties {

    // 对应 date-count: app-OY2tBO7BAk7RtmM2Ij0nPEnF
    private String dateCount;
    
	// 对应 province-count: app-TXjCMb2pC6GQdAHsVqnxx0IU
    private String provinceCount;
    
	// 对应 attack-type-count: app-MiZnAyT7VnsWc7ZBK6xreZd2
    private String attackTypeCount;
}
```

这样就配置完成了，在使用的地方用`@Resource`注解进行注入：

```java
@Resource
private ChaitinApiKeyProperties chaitinApiKeyProperties;

public void function() {
    System.out.println(chaitinApiKeyProperties.getAttackTypeCount())
}
```

