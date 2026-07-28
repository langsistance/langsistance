import '@/styles/landing.css'
import { LandingI18nProvider } from '@/lib/landing-i18n'
import HtmlLangUpdater from '@/components/landing/HtmlLangUpdater'

export default function LandingLayout({ children }: { children: React.ReactNode }) {
  return (
    <LandingI18nProvider>
      <HtmlLangUpdater />
      {children}
    </LandingI18nProvider>
  )
}
