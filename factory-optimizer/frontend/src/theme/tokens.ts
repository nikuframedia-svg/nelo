// Design Tokens - ProdPlan 4.0
// Tema dark preto+verde

export const tokens = {
  colors: {
    background: '#0B0B0B',
    card: {
      default: '#121212',
      elevated: '#161616',
    },
    primary: {
      default: '#00E676',
      hover: '#00C767',
      accent: '#16A34A',
    },
    text: {
      title: '#EAEAEA',
      body: '#CFCFCF',
      secondary: '#9A9A9A',
    },
    danger: '#EF4444',
    warning: '#F59E0B',
    border: '#262626',
  },
  spacing: {
    xs: '8px',
    sm: '12px',
    md: '16px',
    lg: '24px',
    xl: '32px',
  },
  borderRadius: {
    card: '16px',
    input: '16px',
    highlight: '24px',
  },
  typography: {
    fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    fontSize: {
      title: {
        xl: '28px',
        lg: '24px',
        md: '20px',
      },
      body: {
        md: '16px',
        sm: '14px',
      },
    },
    fontWeight: {
      bold: 700,
      semibold: 600,
      regular: 400,
    },
  },
  grid: {
    columns: 12,
    gutter: '24px',
  },
  breakpoints: {
    desktop: '1200px',
    desktopLarge: '1440px',
  },
} as const;

export type Tokens = typeof tokens;

