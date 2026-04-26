import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#ffffff",
        surface: "#f5f5f5",
        stone: "#f5f2ef",
        ink: "#000000",
        muted: "#4e4e4e",
        warm: "#777169",
        border: "#e5e5e5",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        display: ["Inter", "system-ui", "sans-serif"],
      },
      letterSpacing: {
        body: "0.16px",
        bodyLg: "0.18px",
        nav: "0.15px",
        caption: "0.14px",
        displayTight: "-0.96px",
        upper: "0.7px",
      },
      borderRadius: {
        card: "16px",
        section: "24px",
        warm: "30px",
      },
      boxShadow: {
        edge: "rgba(0,0,0,0.075) 0px 0px 0px 0.5px inset",
        outline: "rgba(0,0,0,0.06) 0px 0px 0px 1px, rgba(0,0,0,0.04) 0px 1px 2px, rgba(0,0,0,0.04) 0px 2px 4px",
        elevation: "rgba(0,0,0,0.4) 0px 0px 1px, rgba(0,0,0,0.04) 0px 4px 4px",
        warm: "rgba(78,50,23,0.04) 0px 6px 16px",
      },
    },
  },
  plugins: [],
};

export default config;
