import * as React from "react";

export function Separator({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div 
      className={`h-px w-full bg-border/70 ${className || ""}`} 
      {...props} 
    />
  );
}