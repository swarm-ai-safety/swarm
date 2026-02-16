export function formatNumber(n: number, decimals = 2): string {
  if (Math.abs(n) >= 1000) {
    return n.toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }
  return n.toFixed(decimals);
}

export function formatPercent(n: number, decimals = 1): string {
  return (n * 100).toFixed(decimals) + "%";
}

export function formatCompact(n: number): string {
  if (Math.abs(n) >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (Math.abs(n) >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return n.toFixed(1);
}
