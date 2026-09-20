import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./globals.css";

const DESCRIPTION =
  "A match-3 game, agents that learn to play it, and the finding that a level's difficulty is not one number: it depends on who is holding the controller.";

export const metadata: Metadata = {
  metadataBase: new URL("https://match3-rl.berkaykoklu.com"),
  title: "Difficulty depends on who is playing",
  description: DESCRIPTION,
  openGraph: { title: "Difficulty depends on who is playing", description: DESCRIPTION, type: "website", locale: "en" },
  robots: { index: true, follow: true },
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500&display=swap"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
