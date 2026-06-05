import '@/styles/landing.css'
import { LandingI18nProvider } from '@/lib/landing-i18n'

export default function LandingLayout({ children }: { children: React.ReactNode }) {
  return <LandingI18nProvider>{children}</LandingI18nProvider>
}
