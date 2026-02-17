import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SWARM Isometric Visualization",
  description: "Interactive isometric visualization of multi-agent simulation data",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="bg-bg text-text">{children}</body>
    </html>
  );
}
