"use client"

import { useState } from "react"
import Link from "next/link"
import { Menu, X, MessageCircle, Search, Database, Info } from "lucide-react"
import Logo from "@/components/ui/Logo"

export function Navbar() {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <nav className="bg-primary text-primary-foreground shadow-lg sticky top-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center space-x-3">
            <Logo size={48} className="rounded-md" />
            <div className="text-2xl font-bold">CortexMD</div>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-8">
            <Link
              href="/"
              className="flex items-center space-x-2 hover:text-primary-foreground/80 transition-colors font-medium"
            >
              <Logo size={20} />
              <span>Diagnosis</span>
            </Link>
            <Link
              href="/chat"
              className="flex items-center space-x-2 hover:text-primary-foreground/80 transition-colors font-medium"
            >
              <MessageCircle className="h-4 w-4" />
              <span>AI Chat</span>
            </Link>
            <Link
              href="/medical-knowledge"
              className="flex items-center space-x-2 hover:text-primary-foreground/80 transition-colors font-medium"
            >
              <Search className="h-4 w-4" />
              <span>Knowledge Search</span>
            </Link>
            <Link
              href="/umls"
              className="flex items-center space-x-2 hover:text-primary-foreground/80 transition-colors font-medium"
            >
              <Database className="h-4 w-4" />
              <span>UMLS Lookup</span>
            </Link>
            <Link
              href="/about"
              className="flex items-center space-x-2 hover:text-primary-foreground/80 transition-colors font-medium"
            >
              <Info className="h-4 w-4" />
              <span>About</span>
            </Link>
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden">
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="hover:text-primary-foreground/80 focus:outline-none focus:text-primary-foreground/80"
            >
              {isOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {isOpen && (
          <div className="md:hidden">
            <div className="px-2 pt-2 pb-3 space-y-1 border-t border-primary-foreground/20">
              <Link
                href="/"
                className="flex items-center space-x-2 px-3 py-2 hover:text-primary-foreground/80 hover:bg-primary-foreground/10 rounded-md transition-colors"
                onClick={() => setIsOpen(false)}
              >
                <Logo size={16} />
                <span>Diagnosis</span>
              </Link>
              <Link
                href="/chat"
                className="flex items-center space-x-2 px-3 py-2 hover:text-primary-foreground/80 hover:bg-primary-foreground/10 rounded-md transition-colors"
                onClick={() => setIsOpen(false)}
              >
                <MessageCircle className="h-4 w-4" />
                <span>AI Chat</span>
              </Link>
              <Link
                href="/medical-knowledge"
                className="flex items-center space-x-2 px-3 py-2 hover:text-primary-foreground/80 hover:bg-primary-foreground/10 rounded-md transition-colors"
                onClick={() => setIsOpen(false)}
              >
                <Search className="h-4 w-4" />
                <span>Knowledge Search</span>
              </Link>
              <Link
                href="/umls"
                className="flex items-center space-x-2 px-3 py-2 hover:text-primary-foreground/80 hover:bg-primary-foreground/10 rounded-md transition-colors"
                onClick={() => setIsOpen(false)}
              >
                <Database className="h-4 w-4" />
                <span>UMLS Lookup</span>
              </Link>
              <Link
                href="/about"
                className="flex items-center space-x-2 px-3 py-2 hover:text-primary-foreground/80 hover:bg-primary-foreground/10 rounded-md transition-colors"
                onClick={() => setIsOpen(false)}
              >
                <Info className="h-4 w-4" />
                <span>About</span>
              </Link>
            </div>
          </div>
        )}
      </div>
    </nav>
  )
}
