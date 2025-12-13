import type { Config } from 'tailwindcss'
import { fontFamily } from 'tailwindcss/defaultTheme'

export default {
  darkMode: ['class'],
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    container: {
      center: true,
      padding: '1.5rem',
      screens: {
        '2xl': '1440px',
      },
    },
    extend: {
      colors: {
        background: '#0B0B0B',
        surface: {
          DEFAULT: '#121212',
          elevated: '#161616',
        },
        nikufra: {
          DEFAULT: '#00E676',
          hover: '#00C767',
          accent: '#16A34A',
        },
        text: {
          primary: '#EAEAEA',
          body: '#CFCFCF',
          muted: '#9A9A9A',
        },
        danger: '#EF4444',
        warning: '#F59E0B',
        border: '#262626',
      },
      fontFamily: {
        sans: ['Inter', 'DM Sans', 'Satoshi', ...fontFamily.sans],
        mono: ['"JetBrains Mono"', ...fontFamily.mono],
      },
      borderRadius: {
        xl: '16px',
        '2xl': '24px',
      },
      boxShadow: {
        glow: '0 4px 20px rgba(0, 230, 118, 0.08)',
      },
      keyframes: {
        pulseBorder: {
          '0%, 100%': { boxShadow: '0 0 0 0 rgba(0, 230, 118, 0.35)' },
          '50%': { boxShadow: '0 0 0 8px rgba(0, 230, 118, 0)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-700px 0' },
          '100%': { backgroundPosition: '700px 0' },
        },
      },
      animation: {
        pulseBorder: 'pulseBorder 2s ease-in-out infinite',
        shimmer: 'shimmer 1.5s linear infinite',
      },
    },
  },
  plugins: [],
} satisfies Config
