"""
AI真人剧生成 - 完整流程验证Demo
支持多模型选择
"""

import streamlit as st
import json
import os
from pathlib import Path

# ============== 页面配置 ==============
st.set_page_config(
    page_title="AI真人剧生成验证",
    page_icon="🎥",
    layout="wide"
)

# ============== 模型配置 ==============
MODELS = {
    "Google Gemini 2.5 Pro": {
        "provider": "gemini",
        "model": "gemini-2.5-pro",
        "api_key_name": "GOOGLE_API_KEY",
        "api_url": "https://aistudio.google.com/app/apikey"
    },
    "Google Gemini 2.0 Flash": {
        "provider": "gemini",
        "model": "gemini-2.0-flash",
        "api_key_name": "GOOGLE_API_KEY",
        "api_url": "https://aistudio.google.com/app/apikey"
    },
    "Google Gemini 1.5 Pro": {
        "provider": "gemini",
        "model": "gemini-1.5-pro",
        "api_key_name": "GOOGLE_API_KEY",
        "api_url": "https://aistudio.google.com/app/apikey"
    },
    "OpenAI GPT-4o": {
        "provider": "openai",
        "model": "gpt-4o",
        "api_key_name": "OPENAI_API_KEY",
        "api_url": "https://platform.openai.com/api-keys"
    },
    "OpenAI GPT-4o Mini": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key_name": "OPENAI_API_KEY",
        "api_url": "https://platform.openai.com/api-keys"
    },
}

# ============== LLM调用 ==============
def get_llm_client(model_config: dict, api_key: str):
    """获取LLM客户端"""
    if model_config["provider"] == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model_config["model"])

    elif model_config["provider"] == "openai":
        from openai import OpenAI
        return OpenAI(api_key=api_key)

    return None


def call_llm(client, model_config: dict, prompt: str) -> str:
    """调用LLM"""
    try:
        if model_config["provider"] == "gemini":
            response = client.generate_content(prompt)
            return response.text

        elif model_config["provider"] == "openai":
            response = client.chat.completions.create(
                model=model_config["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"


def parse_json_response(response_text: str) -> dict:
    """解析LLM返回的JSON"""
    try:
        # 尝试提取JSON部分
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0]
        else:
            json_str = response_text

        # 清理可能的markdown标记
        json_str = json_str.strip()
        if json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]

        return json.loads(json_str)
    except Exception as e:
        return {"error": f"解析失败: {str(e)}", "raw": response_text}


def format_json(data: dict) -> str:
    """格式化JSON显示"""
    return json.dumps(data, ensure_ascii=False, indent=2)


# ============== Prompt模板 ==============

# 剧本校验Prompt
SCRIPT_VALIDATION_PROMPT = """
## 任务：校验剧本格式

请校验以下剧本是否满足基本格式要求。

### 校验项（返回JSON格式）：
1. is_script: 是否为剧本（是=true）
2. format_type: 格式类型（"标记说明格式"/"JSON格式"/"非剧本"）
3. issues: 问题列表（如果没有问题则为空数组）
4. summary: 简要说明

### 剧本内容：
{script_content}
"""

# 人物提取Prompt
CHARACTER_EXTRACTION_PROMPT = """
## 任务：从剧本中提取人物信息

请从以下剧本中提取**所有人物信息**。**每个角色都必须单独提取，不能合并！**

### 重要规则

1. **每个角色单独提取**：
   - 主角、配角、反派、龙套都要单独提取
   - ❌ 错误示例："反派角色若干" → 这会合并多个角色
   - ✅ 正确示例：赵大勇、孙健、郑鹏、牛小光 → 每个都是独立角色

2. **多年龄段处理**：
   - 如果同一角色有不同年龄段（幼年/青年/老年），需要拆分为不同版本
   - 格式：`角色名_年龄段`，如 `楚寒烟_幼年`、`楚寒烟_青年`
   - 每个年龄段独立记录appearances

3. **禁止合并**：
   - 绝对不能将多个角色合并为一个"综合角色"
   - 不能使用"其他角色"、"反派若干"、"龙套若干"等模糊描述
   - 每个有名字的角色都要单独列出

### 输出JSON格式：
```json
{{
  "background_setting": {{
    "primary_background": {{"type": "朝代/虚构世界/现实地点", "name": "背景名", "description": "描述"}},
    "special_backgrounds": [],
    "consistency_rule": "一致性规则"
  }},
  "characters": [
    {{
      "id": "char_001",
      "name": "角色名",
      "versions": [
        {{
          "version_id": "char_001_young",
          "version_name": "角色名_青年",
          "age_group": "青年",
          "basic_info": {{
            "gender": "男/女",
            "age": "年龄",
            "face_consistency": {{"face_shape": "脸型", "distinctive_features": "特征", "temperament": "气质"}},
            "appearance": "外貌描述",
            "personality": "性格"
          }},
          "appearances": [
            {{
              "appearance_id": "char_001_young_v1",
              "scene_id": "scene_001",
              "scene_name": "场景名",
              "time_of_day": "日/夜",
              "clothing": "服装",
              "hairstyle": "发型",
              "accessories": "配饰",
              "expression": "表情",
              "pose": "姿态"
            }}
          ],
          "first_appearance": "第1集 第1场"
        }}
      ]
    }}
  ]
}}
```

### 剧本内容：
{script_content}
"""

# 场景提取Prompt
SCENE_EXTRACTION_PROMPT = """
## 任务：从剧本中提取场景信息

请从以下剧本中提取所有场景信息。

### 输出JSON格式：
```json
{{
  "scenes": [
    {{
      "id": "scene_001",
      "name": "场景名",
      "location": "地点",
      "type": "室内/室外",
      "descriptions": [
        {{
          "time_of_day": "日/夜/晨/昏",
          "view_angle": "视角",
          "description": "描述",
          "lighting": "光照",
          "atmosphere": "氛围"
        }}
      ],
      "appears_in": ["第1集 第1场"]
    }}
  ]
}}
```

### 剧本内容：
{script_content}
"""

# 道具提取Prompt
PROP_EXTRACTION_PROMPT = """
## 任务：从剧本中提取道具信息

请从以下剧本中提取所有道具信息。

### 输出JSON格式：
```json
{{
  "props": [
    {{
      "id": "prop_001",
      "name": "道具名",
      "description": "描述",
      "type": "武器/服装/装饰/工具/其他",
      "owner": "使用者",
      "appears_in": ["第1集 第1场"]
    }}
  ]
}}
```

### 剧本内容：
{script_content}
"""

# 分镜提取Prompt
STORYBOARD_PROMPT = """
## 角色设定

你是一位专业的影视导演和分镜师。你的任务是将剧本转换为专业的分镜脚本。

## 重要：多年龄段角色处理
如果剧本中同一角色有不同年龄段的出场（如幼年/青年/老年），需要拆分处理：
- 版本标记格式：`角色名_年龄段`，如 `楚寒烟_青年`、`楚寒烟_幼年`
- 每个年龄段的戏份生成对应的分镜，标记该分镜中角色的年龄段

## 分镜要求

### 镜头拆分规则
- 场景/地点变化 = 新分镜
- 角色登场/退场 = 新分镜
- 对话场景：主镜头 + 角色反应镜头
- 动作描述按动作单元拆分
- 情绪高潮用特写镜头
- 场景建立用远景开场

### 镜头节奏
- 动作场景：2-5秒
- 对话场景：5-10秒
- 情绪场景：8-15秒

### 镜头选择
景别：WS/MS/MCU/CU/ECU/OTS/POV/2S
运镜：固定/推/拉/摇/移/环绕/跟随/升降
角度：平视/仰视/俯视

### 输出JSON格式：
```json
{{
  "project_name": "项目名",
  "episodes": [
    {{
      "episode_number": 1,
      "shots": [
        {{
          "shot_id": "1.1",
          "time_range": "00:00-00:05",
          "duration": 5,
          "description": "分镜画面描述",
          "scene": {{"name": "场景名", "location": "地点", "time_of_day": "日/夜"}},
          "characters": [
            {{
              "name": "角色名",
              "version_name": "角色名_青年",
              "age_group": "青年",
              "position": "画面位置",
              "action": "动作描述",
              "expression": "表情",
              "clothing": "服装描述"
            }}
          ],
          "camera": {{"shot_type": "MS", "movement": "固定", "angle": "平视"}},
          "lighting": {{"type": "自然光", "direction": "左", "mood": "明亮"}},
          "dialogue": {{"type": "对话/旁白/内心OS", "character": "角色名", "content": "台词"}},
          "mood": "情绪",
          "transition": "直接切换"
        }}
      ]
    }}
  ]
}}
```

### 剧本内容：
{script_content}
"""

# 生视频提示词Prompt
VIDEO_PROMPT_PROMPT = """
## 任务：生成生视频提示词

### 输入信息

**分镜脚本**：
{storyboard_json}

**参考图库**：暂无（示例）

**目标模型**：{model_name}

### 要求

1. **提示词结构**：[场景描述] + [角色描述] + [动作描述] + [运镜描述] + [氛围/情绪]

2. **参考图引用规则**：
   - 使用@引用参考图，如：@img_001
   - 参考图已包含的信息不重复描述
   - 提示词简洁，重点描述动作
   - 每个时间段都要引用对应的参考图

3. **{model_name}适配**：
{model_requirements}

### 输出JSON格式：
```json
{{
  "shot_id": "分镜ID",
  "prompt": "生视频提示词",
  "reference_used": ["@img_xxx"],
  "camera_movement": "运镜描述"
}}
```

### 分镜内容：
{shot_content}
"""


# ============== 侧边栏：模型配置 ==============
def render_sidebar():
    """渲染侧边栏"""
    st.sidebar.title("⚙️ 模型配置")

    # 模型选择
    model_name = st.sidebar.selectbox(
        "选择模型",
        options=list(MODELS.keys()),
        index=0
    )

    model_config = MODELS[model_name]

    # API Key输入
    api_key = st.sidebar.text_input(
        f"API Key",
        type="password",
        help=f"请输入{model_name}的API Key"
    )

    # API Key获取链接
    st.sidebar.markdown(f"""
    💡 没有API Key？
    - [获取{model_name} API Key]({model_config['api_url']})
    """)

    return model_config, api_key


def check_api_key(model_config: dict, api_key: str):
    """检查API Key是否有效"""
    if not api_key:
        st.warning("⚠️ 请输入API Key")
        return False

    # 测试连接
    try:
        client = get_llm_client(model_config, api_key)
        # 简单测试调用
        if model_config["provider"] == "gemini":
            response = client.generate_content("Hi")
            return True
        elif model_config["provider"] == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model_config["model"],
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            return True
    except Exception as e:
        st.error(f"❌ API Key验证失败: {str(e)}")
        return False

    return True


# ============== 页面函数 ==============

def page_validation(model_config: dict, api_key: str):
    """剧本校验页面"""
    st.header("📋 剧本校验")

    # 输入
    script_file = st.file_uploader("上传剧本文件", type=['txt', 'md'])
    script_text = st.text_area("或直接输入剧本内容", height=300)

    script_content = ""
    if script_file:
        script_content = script_file.read().decode('utf-8')
    elif script_text:
        script_content = script_text

    if script_content:
        with st.expander("剧本内容预览", expanded=True):
            st.text_area("剧本", script_content, height=150, disabled=True)

        if st.button("开始校验", type="primary", disabled=not api_key):
            # 验证API Key
            if not check_api_key(model_config, api_key):
                return

            with st.spinner("校验中..."):
                client = get_llm_client(model_config, api_key)
                prompt = SCRIPT_VALIDATION_PROMPT.format(script_content=script_content)
                result = call_llm(client, model_config, prompt)

                # 显示原始响应
                with st.expander("原始响应"):
                    st.code(result)

                # 解析并显示结构化结果
                st.subheader("校验结果")
                try:
                    parsed = parse_json_response(result)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("是否为剧本", "✅ 是" if parsed.get("is_script") else "❌ 否")
                    with col2:
                        st.metric("格式类型", parsed.get("format_type", "未知"))

                    if parsed.get("issues"):
                        st.error("问题列表：")
                        for issue in parsed["issues"]:
                            st.write(f"- {issue}")
                    else:
                        st.success("校验通过！")

                    st.json(parsed)
                except Exception as e:
                    st.error(f"解析失败: {e}")


def page_flow_a(model_config: dict, api_key: str):
    """流程A：人物/场景/道具提取"""
    st.header("📦 流程A：信息提取 + 生图")

    # 输入
    script_file = st.file_uploader("上传剧本文件", type=['txt', 'md'])
    script_text = st.text_area("或直接输入剧本内容", height=200)

    script_content = ""
    if script_file:
        script_content = script_file.read().decode('utf-8')
    elif script_text:
        script_content = script_text

    if script_content:
        with st.expander("剧本内容预览", expanded=True):
            st.text_area("剧本", script_content, height=100, disabled=True)

        # 提取按钮
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("提取人物", type="primary", disabled=not api_key):
                if not check_api_key(model_config, api_key):
                    return
                with st.spinner("提取人物中..."):
                    client = get_llm_client(model_config, api_key)
                    prompt = CHARACTER_EXTRACTION_PROMPT.format(script_content=script_content)
                    result = call_llm(client, model_config, prompt)

                    st.subheader("人物提取结果")
                    try:
                        parsed = parse_json_response(result)
                        st.json(parsed, expanded=True)
                        st.session_state['characters'] = parsed
                        st.success(f"成功提取 {len(parsed.get('characters', []))} 个角色")
                    except Exception as e:
                        st.error(f"解析失败: {e}")
                        with st.expander("查看原始响应"):
                            st.code(result)

        with col2:
            if st.button("提取场景", disabled=not api_key):
                if not check_api_key(model_config, api_key):
                    return
                with st.spinner("提取场景中..."):
                    client = get_llm_client(model_config, api_key)
                    prompt = SCENE_EXTRACTION_PROMPT.format(script_content=script_content)
                    result = call_llm(client, model_config, prompt)

                    st.subheader("场景提取结果")
                    try:
                        parsed = parse_json_response(result)
                        st.json(parsed, expanded=True)
                        st.session_state['scenes'] = parsed
                        st.success(f"成功提取 {len(parsed.get('scenes', []))} 个场景")
                    except Exception as e:
                        st.error(f"解析失败: {e}")
                        with st.expander("查看原始响应"):
                            st.code(result)

        with col3:
            if st.button("提取道具", disabled=not api_key):
                if not check_api_key(model_config, api_key):
                    return
                with st.spinner("提取道具中..."):
                    client = get_llm_client(model_config, api_key)
                    prompt = PROP_EXTRACTION_PROMPT.format(script_content=script_content)
                    result = call_llm(client, model_config, prompt)

                    st.subheader("道具提取结果")
                    try:
                        parsed = parse_json_response(result)
                        st.json(parsed, expanded=True)
                        st.session_state['props'] = parsed
                        st.success(f"成功提取 {len(parsed.get('props', []))} 个道具")
                    except Exception as e:
                        st.error(f"解析失败: {e}")
                        with st.expander("查看原始响应"):
                            st.code(result)

        # 显示已提取的结果 - 使用标签页更紧凑展示
        st.divider()
        st.subheader("📋 已提取的结果")

        if 'characters' in st.session_state or 'scenes' in st.session_state or 'props' in st.session_state:
            # 创建标签页
            tab1, tab2, tab3 = st.tabs(["👤 人物", "🏠 场景", "🎭 道具"])

            with tab1:
                if 'characters' in st.session_state:
                    st.json(st.session_state['characters'], expanded=True)
                else:
                    st.info("尚未提取人物")

            with tab2:
                if 'scenes' in st.session_state:
                    st.json(st.session_state['scenes'], expanded=True)
                else:
                    st.info("尚未提取场景")

            with tab3:
                if 'props' in st.session_state:
                    st.json(st.session_state['props'], expanded=True)
                else:
                    st.info("尚未提取道具")
        else:
            st.info("点击上方按钮提取信息")

        # 清除结果按钮
        if st.button("🗑️ 清除已提取结果"):
            st.session_state.pop('characters', None)
            st.session_state.pop('scenes', None)
            st.session_state.pop('props', None)
            st.rerun()


def page_flow_b(model_config: dict, api_key: str):
    """流程B：分镜 + 生视频提示词"""
    st.header("🎬 流程B：分镜 + 生视频提示词")

    # 输入
    script_file = st.file_uploader("上传剧本文件", type=['txt', 'md'])
    script_text = st.text_area("或直接输入剧本内容", height=150)

    script_content = ""
    if script_file:
        script_content = script_file.read().decode('utf-8')
    elif script_text:
        script_content = script_text

    # 选择目标模型
    target_model = st.selectbox("目标视频模型", ["Seedance 2.0", "Wan 2.6"])

    model_requirements = {
        "Seedance 2.0": "动作连贯性好，支持长镜头，中文支持好。动作描述要连贯自然，运镜描述使用常用关键词。",
        "Wan 2.6": "视频质量高，电影感强。强调光影效果和氛围，可以加入光线、色调描述。"
    }

    if script_content:
        with st.expander("剧本内容预览", expanded=True):
            st.text_area("剧本", script_content, height=80, disabled=True)

        # 步骤1：生成分镜
        if st.button("生成分镜", type="primary", disabled=not api_key):
            if not check_api_key(model_config, api_key):
                return

            with st.spinner("生成分镜中..."):
                client = get_llm_client(model_config, api_key)
                prompt = STORYBOARD_PROMPT.format(script_content=script_content)
                result = call_llm(client, model_config, prompt)

                st.subheader("分镜结果")
                try:
                    parsed = parse_json_response(result)
                    st.json(parsed)
                    st.session_state['storyboard'] = parsed
                    st.session_state['shots'] = parsed.get('episodes', [{}])[0].get('shots', [])
                except Exception as e:
                    st.error(f"解析失败: {e}")
                    with st.expander("查看原始响应"):
                        st.code(result)

        # 步骤2：生成生视频提示词
        if 'shots' in st.session_state and st.session_state['shots']:
            st.divider()
            st.subheader("🎥 生视频提示词生成")

            # 选择要生成提示词的分镜
            shot_options = [f"{s.get('shot_id', 'N/A')}: {s.get('description', '')[:50]}..."
                           for s in st.session_state['shots']]
            selected_shot_idx = st.selectbox("选择分镜", range(len(shot_options)), format_func=lambda x: shot_options[x])

            # 显示选中的分镜详情
            with st.expander("分镜详情"):
                st.json(st.session_state['shots'][selected_shot_idx])

            # 生成提示词
            if st.button("生成提示词", disabled=not api_key):
                if not check_api_key(model_config, api_key):
                    return

                shot = st.session_state['shots'][selected_shot_idx]

                with st.spinner("生成中..."):
                    client = get_llm_client(model_config, api_key)
                    prompt = VIDEO_PROMPT_PROMPT.format(
                        storyboard_json=json.dumps(st.session_state['storyboard'], ensure_ascii=False),
                        reference_images="暂无（示例）",
                        model_name=target_model,
                        model_requirements=model_requirements[target_model],
                        shot_content=json.dumps(shot, ensure_ascii=False)
                    )
                    result = call_llm(client, model_config, prompt)

                    st.subheader("生视频提示词")
                    try:
                        parsed = parse_json_response(result)
                        st.json(parsed)

                        # 显示格式化后的提示词
                        if 'prompt' in parsed:
                            st.subheader("提示词预览")
                            st.success(parsed['prompt'])
                    except Exception as e:
                        st.error(f"解析失败: {e}")
                        with st.expander("查看原始响应"):
                            st.code(result)

        # 清除分镜结果
        if 'storyboard' in st.session_state:
            if st.button("🗑️ 清除分镜结果"):
                st.session_state.pop('storyboard', None)
                st.session_state.pop('shots', None)
                st.rerun()


def page_home():
    """首页"""
    st.title("🎥 AI真人剧生成 - 验证Demo")
    st.markdown("""
    ## 完整流程验证

    本Demo用于验证AI真人剧生成的完整流程：

    1. **剧本校验** - 验证剧本格式是否正确
    2. **流程A** - 提取人物/场景/道具，生成参考图
    3. **流程B** - 生成分镜，生成视频提示词

    ## 使用说明

    1. 在左侧配置API Key（支持Google Gemini、OpenAI）
    2. 选择要验证的环节
    3. 上传剧本文件或直接输入剧本内容
    4. 点击按钮执行相应操作
    5. 查看每个节点的输入/输出JSON

    ## 示例剧本

    ```
    第1集：穷书生进城
    1-1 日 外 城门口
    人物：苏源

    ▲ 苏源骑驴进城，伸懒腰。
    ▲ 驴嘶鸣，兴奋蹬蹄。
    苏源：（内心OS）十年了，终于进城了。
    ```
    """)

    # 显示支持的模型
    st.subheader("支持的模型")
    st.markdown("| 模型 | 提供商 |")
    st.markdown("|------|--------|")
    for name, config in MODELS.items():
        st.markdown(f"| {name} | {config['provider']} |")


# ============== 主程序 ==============

def main():
    # 渲染侧边栏
    model_config, api_key = render_sidebar()

    # 侧边栏导航
    st.sidebar.title("📑 导航")
    page = st.sidebar.radio(
        "选择环节",
        ["首页", "剧本校验", "流程A：信息提取", "流程B：分镜+生视频"]
    )

    # 根据选择显示不同页面
    if page == "首页":
        page_home()
    elif page == "剧本校验":
        page_validation(model_config, api_key)
    elif page == "流程A：信息提取":
        page_flow_a(model_config, api_key)
    elif page == "流程B：分镜+生视频":
        page_flow_b(model_config, api_key)


if __name__ == "__main__":
    main()
