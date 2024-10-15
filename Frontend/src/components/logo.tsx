import React from "react";
import { cn } from "@/lib/utils";

interface DeepShieldLogoProps {
  className?: string;
  size?: "sm" | "md" | "lg" | "xl";
  showText?: boolean;
}

export function DeepShieldLogo({
  className,
  size = "md",
  showText = false,
}: DeepShieldLogoProps) {
  const sizeMap = {
    sm: "size-7",
    md: "size-9",
    lg: "size-12",
    xl: "size-16",
  };

  return (
    <div className={cn("inline-flex items-center gap-2.5", className)}>
      <div
        className={cn(
          "relative grid place-items-center rounded-xl overflow-hidden p-0.5 border border-cyan/40 bg-black/80 shadow-[0_0_20px_rgba(77,208,225,0.3)] transition-transform duration-300 hover:scale-105",
          sizeMap[size]
        )}
      >
        <img
          src="/deepshield_logo.png"
          alt="DeepShield-X Logo"
          className="h-full w-full object-cover rounded-lg"
          onError={(e) => {
            // Fallback SVG if image not loaded
            e.currentTarget.style.display = "none";
          }}
        />
      </div>
      {showText && (
        <div className="flex flex-col leading-none">
          <span className="font-mono text-sm font-bold tracking-wider text-cyan">
            DEEPSHIELD-X
          </span>
          <span className="font-mono text-[9px] text-muted-foreground uppercase tracking-widest mt-0.5">
            Neural Forensic Lab
          </span>
        </div>
      )}
    </div>
  );
}

export default DeepShieldLogo;
