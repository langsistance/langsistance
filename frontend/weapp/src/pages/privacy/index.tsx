import { ScrollView, Text, View } from '@tarojs/components'
import './index.scss'

/**
 * 小程序隐私政策。
 * 与 web 的 /privacy-policy 是两份独立文本——那份写的是 Chrome 扩展权限，
 * 语境不同，不可混用。
 */
export default function PrivacyPage() {
  return (
    <ScrollView className='privacy-page' scrollY>
      <View className='privacy-doc'>
        <Text className='privacy-doc-title'>CopiioAI 专利助手隐私政策</Text>
        <Text className='privacy-doc-date'>更新日期：2026-09-11</Text>

        <Text className='privacy-h'>1. 我们收集哪些信息</Text>
        <Text className='privacy-p'>
          1.1 微信登录标识。你使用微信一键登录时，我们会通过微信提供的登录凭证
          （code）换取你的微信用户唯一标识（openid），用于识别你的账号并关联你的
          历史对话。我们不会获取你的微信昵称、头像或手机号。
        </Text>
        <Text className='privacy-p'>
          1.2 你主动提交的内容。包括你输入的提问文字，以及你从微信会话中选择上传的
          专利文档（PDF / DOCX / XML）。这些内容用于执行你请求的检索与分析。
        </Text>
        <Text className='privacy-p'>
          1.3 对话记录。你的提问与相应的分析结果会保存为历史会话，以便你随时回看。
        </Text>

        <Text className='privacy-h'>2. 我们如何使用这些信息</Text>
        <Text className='privacy-p'>
          仅用于提供专利检索、专利分析与相关结果导出服务，以及维护你的历史会话。
        </Text>

        <Text className='privacy-h'>3. 信息的存储与保留</Text>
        <Text className='privacy-p'>
          上述信息存储在我们的服务器上。你可以在小程序内删除任意历史会话，删除后该
          会话不再展示。
        </Text>

        <Text className='privacy-h'>4. 信息的共享</Text>
        <Text className='privacy-p'>
          我们不会将你的个人信息出售或提供给第三方用于营销目的。为完成你请求的分析，
          提问内容会发送至第三方大语言模型服务处理。
        </Text>

        <Text className='privacy-h'>5. 你的权利</Text>
        <Text className='privacy-p'>
          你可以随时在小程序内删除历史会话；如需注销账号或删除全部数据，可通过下方
          联系方式与我们联系。
        </Text>

        <Text className='privacy-h'>6. 联系我们</Text>
        <Text className='privacy-p'>copiioai.com@gmail.com</Text>
      </View>
    </ScrollView>
  )
}
