export function validateSignupPasswordConfirmation(password, confirmPassword, lang = 'en') {
  if (password !== confirmPassword) {
    return lang === 'en' ? 'Passwords do not match' : '两次输入的密码不一致'
  }
  return ''
}
