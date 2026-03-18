# 验证Demo - AI真人剧生成

基于Streamlit的完整流程验证Demo。

## 功能

1. **剧本校验** - 验证剧本格式
2. **流程A** - 提取人物/场景/道具
3. **流程B** - 生成分镜，生成视频提示词

## 特点

- 支持多模型选择（Google Gemini、OpenAI）
- 界面输入API Key，无需配置环境变量
- 方便分享给他人试用

## 环境准备

```bash
pip install streamlit openai google-generativeai
```

## 运行

```bash
streamlit run app.py
```

运行后打开浏览器 http://localhost:8501

## 使用说明

1. 在左侧选择模型（Google Gemini 或 OpenAI）
2. 输入API Key
3. 选择要验证的环节
4. 上传剧本或直接输入剧本内容
5. 点击按钮执行操作，查看结果JSON
