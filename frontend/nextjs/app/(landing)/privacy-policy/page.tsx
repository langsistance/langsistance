import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'Privacy Policy',
  description: 'Privacy Policy for CopiioAI — how we collect, use, and protect your data.',
  alternates: {
    canonical: 'https://copiioai.com/privacy-policy',
  },
}

export default function PrivacyPolicy() {
  return (
    <div className="max-w-3xl mx-auto px-6 py-16">
      <nav className="mb-5 text-sm text-gray-500">
        <a href="/" className="text-blue-600 hover:underline">Home</a>
        {' / '}
        <span>Privacy Policy</span>
      </nav>

      <h1 className="text-3xl font-bold text-gray-900 mb-2">Privacy Policy for CopiioAI</h1>
      <p className="text-gray-500 mb-8">Last updated: 2026-03-05.</p>

      <p className="text-gray-700 mb-6">
        CopiioAI ("we", "our", or "us") is an AI-powered developer tool designed to help users
        turn APIs and web requests into reusable AI tools and knowledge.
        We respect your privacy and are committed to handling data transparently and responsibly.
      </p>

      <h2 className="text-2xl font-bold text-gray-900 mt-10 mb-4">1. Scope and Purpose</h2>
      <p className="text-gray-700 mb-4">
        This Privacy Policy explains how the CopiioAI Chrome extension collects, uses, and protects
        information when you use the product.
        CopiioAI is a productivity and developer utility — it is <strong>not</strong> an advertising,
        analytics, or tracking service.
      </p>

      <h2 className="text-2xl font-bold text-gray-900 mt-10 mb-4">2. Information We Collect</h2>

      <h3 className="text-xl font-semibold text-gray-900 mt-6 mb-3">2.1 User-Provided Content</h3>
      <p className="text-gray-700 mb-3">We collect data that you explicitly provide when using CopiioAI, including:</p>
      <ul className="list-disc pl-6 text-gray-700 mb-4 space-y-1">
        <li>API definitions (e.g. JSON)</li>
        <li>User-created tools and instructions ("knowledge")</li>
        <li>Configuration settings and preferences</li>
        <li>Conversation inputs and prompts</li>
      </ul>
      <p className="text-gray-700 mb-4">
        This content is created intentionally by the user and used solely to enable CopiioAI's core functionality.
      </p>

      <h3 className="text-xl font-semibold text-gray-900 mt-6 mb-3">2.2 Authentication Information</h3>
      <p className="text-gray-700 mb-4">
        If you sign in, CopiioAI collects the minimum identity information required to authenticate
        and associate data with your account, such as a secure identity token.
        We do not access contacts, email content, or unrelated profile data.
      </p>

      <h3 className="text-xl font-semibold text-gray-900 mt-6 mb-3">2.3 Network Request Data (User-Initiated Only)</h3>
      <p className="text-gray-700 mb-3">
        When you explicitly enable developer mode or browser capture mode, CopiioAI may temporarily
        access network request data, including:
      </p>
      <ul className="list-disc pl-6 text-gray-700 mb-4 space-y-1">
        <li>Request URLs</li>
        <li>HTTP methods</li>
        <li>Headers and request payloads related to APIs</li>
      </ul>
      <p className="text-gray-700 mb-4">
        This data is accessed <strong>only</strong> to allow users to convert selected HTTP requests
        into AI-usable tools. CopiioAI does not continuously monitor browsing activity.
      </p>

      <h3 className="text-xl font-semibold text-gray-900 mt-6 mb-3">2.4 Analytics and Usage Data</h3>
      <p className="text-gray-700 mb-3">
        CopiioAI uses Google Analytics 4 to collect anonymous usage statistics that help us
        understand how the product is used and improve user experience. This may include:
      </p>
      <ul className="list-disc pl-6 text-gray-700 mb-4 space-y-1">
        <li>Page views and feature usage patterns</li>
        <li>Device type, browser version, and operating system</li>
        <li>Geographic location (country/region level only)</li>
        <li>Error reports and performance metrics</li>
      </ul>
      <p className="text-gray-700 mb-4">
        Google Analytics data is anonymized and does not contain personally identifiable information
        or user-generated content. We use this data solely for product improvement and do not share
        it with third parties for advertising purposes.
      </p>
      <p className="text-gray-700 mb-4">
        You can opt out of Google Analytics tracking through your browser settings or by using
        browser extensions like the Google Analytics Opt-out Browser Add-on.
      </p>

      <h2 className="text-2xl font-bold text-gray-900 mt-10 mb-4">3. Cookies and Authenticated Requests</h2>
      <p className="text-gray-700 mb-3">CopiioAI may access cookies for user-selected websites when necessary to:</p>
      <ul className="list-disc pl-6 text-gray-700 mb-4 space-y-1">
        <li>Capture authenticated API requests</li>
        <li>Verify that APIs function correctly</li>
      </ul>
      <p className="text-gray-700 mb-4">
        Cookies are accessed only during explicit user actions, are not used for tracking,
        and are not stored or reused beyond the session.
      </p>

      <h2 className="text-2xl font-bold text-gray-900 mt-10 mb-4">4. How We Use Information</h2>
      <p className="text-gray-700 mb-3">Collected information is used exclusively to:</p>
      <ul className="list-disc pl-6 text-gray-700 mb-4 space-y-1">
        <li>Provide and operate CopiioAI's core features</li>
        <li>Enable API capture, conversion, and execution</li>
        <li>Support AI-powered conversations using user-defined tools</li>
        <li>Allow users to share or authorize access to tools and knowledge</li>
        <li>Improve reliability and user experience</li>
        <li>Analyze anonymous usage patterns through Google Analytics to identify bugs and optimize features</li>
      </ul>
      <p className="text-gray-700 mb-4">
        CopiioAI does <strong>not</strong> use data for advertising, tracking, or profiling.
        Analytics data is anonymized and used solely for product improvement.
      </p>

      <h2 className="text-2xl font-bold text-gray-900 mt-10 mb-4">5. Browser Permissions</h2>
      <p className="text-gray-700 mb-3">CopiioAI requests browser permissions strictly to support its stated functionality:</p>
      <ul className="list-disc pl-6 text-gray-700 mb-4 space-y-1">
        <li><strong>storage</strong> – store user settings and configurations</li>
        <li><strong>tabs &amp; webNavigation</strong> – interact with active tabs during user-initiated capture</li>
        <li><strong>webRequest &amp; declarativeNetRequest</strong> – capture and analyze selected HTTP requests</li>
        <li><strong>scripting</strong> – inject scripts when explicitly enabled by the user</li>
        <li><strong>identity</strong> – support secure authentication</li>
        <li><strong>offscreen</strong> – perform background processing related to capture and analysis</li>
        <li><strong>cookies</strong> – access authentication cookies when required for API capture; also used by Google Analytics for anonymous usage tracking</li>
      </ul>
      <p className="text-gray-700 mb-4">
        The <strong>&lt;all_urls&gt;</strong> host permission is required because CopiioAI is a general-purpose
        developer tool that must operate on user-selected websites and APIs.
        CopiioAI does not scan or monitor all websites by default.
      </p>

      <h2 className="text-2xl font-bold text-gray-900 mt-10 mb-4">6. Data Sharing</h2>
      <p className="text-gray-700 mb-3">CopiioAI does not sell or rent user data. Data may be shared only:</p>
      <ul className="list-disc pl-6 text-gray-700 mb-4 space-y-1">
        <li>When you explicitly share tools or knowledge with other users</li>
        <li>To provide core service functionality</li>
        <li>When required by law</li>
      </ul>

      <h2 className="text-2xl font-bold text-gray-900 mt-10 mb-4">7. Third-Party Services</h2>
      <p className="text-gray-700 mb-4">CopiioAI uses the following third-party services to operate and improve the product:</p>

      <h3 className="text-xl font-semibold text-gray-900 mt-6 mb-3">7.1 Google Analytics 4</h3>
      <p className="text-gray-700 mb-4">
        We use Google Analytics 4 to collect and analyze anonymous usage statistics.
        Google Analytics may set cookies on your device to track usage patterns across sessions.
        Google's use of data is governed by their Privacy Policy, available at{' '}
        <a href="https://policies.google.com/privacy" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
          https://policies.google.com/privacy
        </a>.
      </p>
      <p className="text-gray-700 mb-3">Google Analytics collects data such as:</p>
      <ul className="list-disc pl-6 text-gray-700 mb-4 space-y-1">
        <li>How you interact with CopiioAI features</li>
        <li>Time spent using the extension</li>
        <li>Technical information about your device and browser</li>
        <li>Approximate geographic location (IP-based, anonymized)</li>
      </ul>
      <p className="text-gray-700 mb-4">
        This data helps us understand feature usage, identify bugs, and improve the product.{' '}
        <strong>No user-created content</strong> (such as API definitions, tools, or conversation data)
        is sent to Google Analytics.
      </p>

      <h2 className="text-2xl font-bold text-gray-900 mt-10 mb-4">8. Remote Code Execution</h2>
      <p className="text-gray-700 mb-4">
        CopiioAI does <strong>not</strong> download or execute remote code.
        All executable logic is bundled with the extension at install time.
      </p>

      <h2 className="text-2xl font-bold text-gray-900 mt-10 mb-4">9. Data Retention and Security</h2>
      <p className="text-gray-700 mb-4">
        We retain data only as long as necessary to provide the service.
        Temporary capture data is discarded after use.
        We apply reasonable technical and organizational measures to protect user data.
      </p>

      <h2 className="text-2xl font-bold text-gray-900 mt-10 mb-4">10. User Control</h2>
      <p className="text-gray-700 mb-3">You control when CopiioAI accesses data:</p>
      <ul className="list-disc pl-6 text-gray-700 mb-4 space-y-1">
        <li>Capture and developer modes are opt-in</li>
        <li>Sharing and authorization are user-controlled</li>
        <li>You may delete your data or revoke access at any time</li>
      </ul>

      <h2 className="text-2xl font-bold text-gray-900 mt-10 mb-4">11. Changes to This Policy</h2>
      <p className="text-gray-700 mb-4">
        We may update this Privacy Policy periodically.
        Any changes will be reflected with an updated "Last updated" date.
      </p>

      <h2 className="text-2xl font-bold text-gray-900 mt-10 mb-4">12. Contact</h2>
      <p className="text-gray-700 mb-2">If you have questions about this Privacy Policy, please contact:</p>
      <p className="text-gray-700">
        <strong>Email:</strong> copiioai.com@gmail.com<br />
        <strong>Website:</strong> https://copiioai.com/privacy-policy
      </p>
    </div>
  )
}
