// Permission system for RSI — inspired by opencode and codex CLI's permission models.
// Provides granular control over which tools require approval, are allowed, or are denied.
// Configured in ~/.config/rsi/config.json under the "permissions" key.

export type PermissionEffect = "allow" | "ask" | "deny"

export interface PermissionRule {
  /** Tool name or pattern (e.g. "run_command", "run_command:rm *", "*"). */
  pattern: string
  /** The effect when this rule matches. */
  effect: PermissionEffect
}

export interface PermissionConfig {
  /** Default effect when no rule matches. */
  default: PermissionEffect
  /** Ordered rules — last matching rule wins (like opencode). */
  rules: PermissionRule[]
}

export const DEFAULT_PERMISSIONS: PermissionConfig = {
  default: "allow",
  rules: [
    // Destructive operations require approval
    { pattern: "run_command:rm -rf *", effect: "ask" },
    { pattern: "run_command:sudo *", effect: "ask" },
    { pattern: "run_command:git push *", effect: "ask" },
    { pattern: "run_command:git reset --hard *", effect: "ask" },
    { pattern: "run_command:git checkout -- *", effect: "ask" },
    { pattern: "run_command:git clean -fd *", effect: "ask" },
    { pattern: "run_command:git commit --no-verify *", effect: "deny" },
    { pattern: "delete_directory", effect: "ask" },
    { pattern: "delete_file", effect: "ask" },
    // File writes to sensitive paths
    { pattern: "write_file:.env*", effect: "ask" },
    { pattern: "edit_file:.env*", effect: "ask" },
    { pattern: "write_file:**/.ssh/**", effect: "deny" },
    { pattern: "edit_file:**/.ssh/**", effect: "deny" },
  ],
}

/** Check if a tool call is permitted. Returns the effect (allow/ask/deny). */
export function checkPermission(
  toolName: string,
  args: Record<string, unknown>,
  config: PermissionConfig = DEFAULT_PERMISSIONS,
): PermissionEffect {
  // Build the string to match against
  // For run_command, include the command text: "run_command:git push origin main"
  // For file tools, include the path: "write_file:.env"
  let matchStr = toolName
  if (toolName === "run_command" && args.command) {
    matchStr = `${toolName}:${String(args.command)}`
  } else if (args.path) {
    matchStr = `${toolName}:${String(args.path)}`
  } else if (args.source) {
    matchStr = `${toolName}:${String(args.source)}`
  }

  let effect = config.default
  for (const rule of config.rules) {
    if (matchesPattern(matchStr, rule.pattern)) {
      effect = rule.effect
    }
  }
  return effect
}

/** Simple glob pattern matcher supporting * and ** wildcards. */
function matchesPattern(str: string, pattern: string): boolean {
  // Convert glob to regex
  let regex = "^"
  let i = 0
  while (i < pattern.length) {
    const c = pattern[i]
    if (c === "*" && pattern[i + 1] === "*") {
      regex += ".*"
      i += 2
    } else if (c === "*") {
      regex += "[^]*"
      i++
    } else if ("\\.+?^${}()|[]".includes(c)) {
      regex += "\\" + c
      i++
    } else {
      regex += c
      i++
    }
  }
  regex += "$"
  try {
    return new RegExp(regex, "i").test(str)
  } catch {
    return str === pattern
  }
}

/** Load permission config from a parsed config object. */
export function loadPermissions(raw: any): PermissionConfig {
  if (!raw || typeof raw !== "object") return DEFAULT_PERMISSIONS
  return {
    default: raw.default === "ask" || raw.default === "deny" ? raw.default : "allow",
    rules: Array.isArray(raw.rules)
      ? raw.rules
          .filter((r: any) => r && r.pattern && r.effect)
          .map((r: any) => ({ pattern: String(r.pattern), effect: r.effect as PermissionEffect }))
      : DEFAULT_PERMISSIONS.rules,
  }
}
