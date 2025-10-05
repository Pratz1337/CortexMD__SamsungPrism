"use client"

import Image from "next/image"
import * as React from "react"

export interface LogoProps {
  size?: number
  className?: string
  alt?: string
}

export function Logo({ size = 48, className = "", alt = "CortexMD" }: LogoProps) {
  return (
    <Image
      src="/logo.png"
      alt={alt}
      width={size}
      height={size}
      className={className}
      priority
    />
  )
}

export default Logo
