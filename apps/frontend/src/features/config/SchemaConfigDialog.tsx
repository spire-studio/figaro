import { useEffect, useMemo, useState, type ReactNode } from "react";

import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "../../components/ui/alert-dialog";
import { Button } from "../../components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../../components/ui/tabs";
import type { NodeRole } from "../../pages/types";
import { getValueByPath, isFieldDefinition, isRecord } from "../simulation/utils";

type SchemaRenderFn = (
  schemaSection: Record<string, unknown>,
  pathPrefix: string,
  role: NodeRole,
  properties: Record<string, unknown>,
  onChangeProperty: (path: string, value: unknown) => void,
) => ReactNode[];

type SchemaConfigDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description?: string;
  busy?: boolean;
  schema: Record<string, unknown> | null;
  role: NodeRole;
  properties: Record<string, unknown> | null;
  onChangeProperty: (path: string, value: unknown) => void;
  renderSchemaSection: SchemaRenderFn;
  onSave: () => void;
  saveLabel?: string;
  beforeSchema?: ReactNode;
  maxWidthClassName?: string;
  schemaMissingText?: string;
};

function checkDependency(properties: Record<string, unknown>, dependsOn?: string): boolean {
  if (!dependsOn) return true;

  return dependsOn.split("&&").every((condition) => {
    const match = condition.trim().match(/(.+)\s*==\s*(.+)/);
    if (!match) return true;
    const actual = getValueByPath(properties, match[1].trim());

    const rawTarget = match[2].trim().replace(/^['"]|['"]$/g, "");
    if (rawTarget === "true") return Boolean(actual) === true;
    if (rawTarget === "false") return Boolean(actual) === false;
    const numeric = Number(rawTarget);
    if (Number.isFinite(numeric)) return Number(actual) === numeric;
    return String(actual) === rawTarget;
  });
}

export function SchemaConfigDialog({
  open,
  onOpenChange,
  title,
  description,
  busy = false,
  schema,
  role,
  properties,
  onChangeProperty,
  renderSchemaSection,
  onSave,
  saveLabel = "Save Config",
  beforeSchema,
  maxWidthClassName = "max-w-5xl",
  schemaMissingText = "Config schema is not loaded.",
}: SchemaConfigDialogProps) {
  const sections = useMemo(() => {
    if (!schema || !properties) return [];

    return Object.entries(schema)
      .filter(([key, rawValue]) => {
        if (key === "role" || key === "depends_on" || key === "hidden" || key === "ui") return false;
        if (!isRecord(rawValue) || isFieldDefinition(rawValue)) return false;

        const roleTag = typeof rawValue.role === "string" ? rawValue.role : null;
        const hidden = rawValue.hidden === true;
        const dependsOn = typeof rawValue.depends_on === "string" ? rawValue.depends_on : undefined;
        if (hidden) return false;
        if (roleTag && roleTag !== role && roleTag !== "all") return false;
        return checkDependency(properties, dependsOn);
      })
      .map(([key, value]) => {
        const fields = renderSchemaSection(
          value as Record<string, unknown>,
          key,
          role,
          properties,
          onChangeProperty,
        );
        return {
          key,
          label: key
            .split("_")
            .map((part) => (part.length > 0 ? `${part[0].toUpperCase()}${part.slice(1)}` : part))
            .join(" "),
          fields,
        };
      })
      .filter((section) => section.fields.length > 0);
  }, [schema, role, properties, onChangeProperty, renderSchemaSection]);

  const [activeSection, setActiveSection] = useState("");
  useEffect(() => {
    if (sections.length === 0) {
      setActiveSection("");
      return;
    }
    if (!sections.some((section) => section.key === activeSection)) {
      setActiveSection(sections[0].key);
    }
  }, [sections, activeSection]);

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent className={maxWidthClassName}>
        <AlertDialogHeader>
          <AlertDialogTitle>{title}</AlertDialogTitle>
          {description && <AlertDialogDescription>{description}</AlertDialogDescription>}
        </AlertDialogHeader>

        {(!schema || !properties) && <p className="text-sm text-muted-foreground">{schemaMissingText}</p>}
        {schema && properties && (
          <div className="space-y-3 [&_input]:font-mono [&_select]:font-mono [&_textarea]:font-mono">
            {beforeSchema}
            {sections.length === 0 && <p className="text-sm text-muted-foreground">No editable config sections.</p>}
            {sections.length > 0 && activeSection && (
              <Tabs value={activeSection} onValueChange={setActiveSection}>
                <div className="grid gap-3 grid-cols-[180px_minmax(0,1fr)] md:grid-cols-[220px_minmax(0,1fr)]">
                  <div className="h-[56vh] overflow-auto rounded-md border border-border/70 bg-muted/20 p-2">
                    <TabsList className="h-auto w-full flex-col items-stretch justify-start gap-1 bg-transparent p-0 text-foreground">
                      {sections.map((section) => (
                        <TabsTrigger key={section.key} value={section.key} className="w-full justify-start px-3 py-1.5">
                          {section.label}
                        </TabsTrigger>
                      ))}
                    </TabsList>
                  </div>

                  <div className="h-[56vh]">
                    {sections.map((section) => (
                      <TabsContent key={section.key} value={section.key} className="mt-0 h-full">
                        <div className="h-full space-y-2 overflow-auto rounded-md border p-3">{section.fields}</div>
                      </TabsContent>
                    ))}
                  </div>
                </div>
              </Tabs>
            )}
          </div>
        )}

        <AlertDialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={onSave} disabled={busy}>
            {saveLabel}
          </Button>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
