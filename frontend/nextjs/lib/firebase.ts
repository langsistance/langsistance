/**
 * Firebase JS SDK initialization — auth only.
 *
 * Used by the Google sign-in flow.  The SDK talks to Firebase directly from the
 * browser (signInWithPopup opens a Google OAuth popup), then the resulting
 * Google idToken is exchanged server‑side via POST /auth/google so the backend
 * can return a Firebase refreshToken as well.
 *
 * Config values can be overridden via NEXT_PUBLIC_FIREBASE_* env vars.
 */
import { initializeApp, type FirebaseApp } from 'firebase/app'
import { getAuth, type Auth } from 'firebase/auth'

const firebaseConfig = {
  apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY || 'AIzaSyDEiHxA2Ml1ZbUF1xuR0281zyrCcIUnRzU',
  authDomain: process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN || 'langsistance.firebaseapp.com',
  projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID || 'langsistance',
  appId: process.env.NEXT_PUBLIC_FIREBASE_APP_ID || '1:1063325627344:web:c664ff01e37688bfe33fd5',
}

let app: FirebaseApp | null = null
let auth: Auth | null = null

export function getFirebaseApp(): FirebaseApp {
  if (!app) {
    app = initializeApp(firebaseConfig)
  }
  return app
}

export function getFirebaseAuth(): Auth {
  if (!auth) {
    auth = getAuth(getFirebaseApp())
  }
  return auth
}
