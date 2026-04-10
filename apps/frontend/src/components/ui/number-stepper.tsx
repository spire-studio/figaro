import * as React from "react";
import { Minus, Plus } from "lucide-react";

import { cn } from "../../lib/utils";
import { Input } from "./input";

type NumberStepperProps = {
  value: number;
  onValueChange: (next: number) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
  className?: string;
  inputClassName?: string;
};

function clampNumber(value: number, min?: number, max?: number): number {
  let next = value;
  if (min !== undefined) next = Math.max(min, next);
  if (max !== undefined) next = Math.min(max, next);
  return next;
}

function normalizeStep(step?: number): number {
  if (step === undefined || !Number.isFinite(step) || step <= 0) return 1;
  return step;
}

function stepDecimals(step: number): number {
  const text = String(step);
  const dot = text.indexOf(".");
  if (dot < 0) return 0;
  return text.length - dot - 1;
}

export function NumberStepper({
  value,
  onValueChange,
  min,
  max,
  step,
  disabled = false,
  className,
  inputClassName,
}: NumberStepperProps) {
  const safeStep = normalizeStep(step);
  const decimals = stepDecimals(safeStep);
  const current = Number.isFinite(value) ? value : 0;

  function commit(next: number): void {
    const rounded = Number(next.toFixed(decimals));
    onValueChange(clampNumber(rounded, min, max));
  }

  return (
    <div className={cn("flex items-center gap-2", className)}>
      <Input
        type="number"
        value={current}
        min={min}
        max={max}
        step={safeStep}
        disabled={disabled}
        onChange={(event) => {
          const parsed = Number(event.target.value);
          if (!Number.isFinite(parsed)) return;
          commit(parsed);
        }}
        className={cn(
          "text-center [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none",
          inputClassName,
        )}
      />
      <div className="inline-flex overflow-hidden rounded-md border border-input bg-background/60">
        <button
          type="button"
          className="inline-flex h-9 w-9 items-center justify-center border-r border-input text-foreground transition hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50"
          onClick={() => commit(current - safeStep)}
          disabled={disabled}
          aria-label="Decrease value"
        >
          <Minus className="h-4 w-4" />
        </button>
        <button
          type="button"
          className="inline-flex h-9 w-9 items-center justify-center text-foreground transition hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50"
          onClick={() => commit(current + safeStep)}
          disabled={disabled}
          aria-label="Increase value"
        >
          <Plus className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}
