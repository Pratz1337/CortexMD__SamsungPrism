/**
 * Debounce utility to prevent excessive API calls
 */

export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null
  
  return function executedFunction(...args: Parameters<T>) {
    const later = () => {
      timeout = null
      func(...args)
    }
    
    if (timeout) {
      clearTimeout(timeout)
    }
    
    timeout = setTimeout(later, wait)
  }
}

export function throttle<T extends (...args: any[]) => any>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle: boolean = false
  
  return function executedFunction(this: any, ...args: Parameters<T>) {
    if (!inThrottle) {
      func.apply(this, args)
      inThrottle = true
      setTimeout(() => inThrottle = false, limit)
    }
  }
}

// Cache for API responses to prevent duplicate requests
const apiCache = new Map<string, { data: any; timestamp: number }>()
const CACHE_TTL = 30000 // 30 seconds for production optimization
const PATIENT_DATA_CACHE_TTL = 60000 // 1 minute for patient data

export function getCachedResponse(key: string): any | null {
  const cached = apiCache.get(key)
  const ttl = key.includes('patient') ? PATIENT_DATA_CACHE_TTL : CACHE_TTL
  
  if (cached && Date.now() - cached.timestamp < ttl) {
    return cached.data
  }
  apiCache.delete(key)
  return null
}

export function setCachedResponse(key: string, data: any): void {
  apiCache.set(key, { data, timestamp: Date.now() })
}

// Optimized for single retrieval per patient click
export function isCacheValid(key: string): boolean {
  const cached = apiCache.get(key)
  const ttl = key.includes('patient') ? PATIENT_DATA_CACHE_TTL : CACHE_TTL
  return Boolean(cached && (Date.now() - cached.timestamp < ttl))
}

export function clearCache(): void {
  apiCache.clear()
}