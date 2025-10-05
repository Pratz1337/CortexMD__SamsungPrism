"use client";

import * as React from "react";
import { Heart, Activity } from "lucide-react";
import Logo from "@/components/ui/Logo"
import { motion, AnimatePresence } from "framer-motion";

interface HealthcareLoadingScreenProps {
  variant?: "heartbeat" | "stethoscope" | "pulse";
  message?: string;
  className?: string;
}

function cn(...classes: (string | undefined | null | boolean)[]): string {
  return classes.filter(Boolean).join(' ');
}

const HeartbeatLoader = ({ className }: { className?: string }) => {
  return (
    <div className={cn("flex flex-col items-center space-y-6", className)}>
      <div className="relative">
        <motion.div
          animate={{
            scale: [1, 1.2, 1],
          }}
          transition={{
            duration: 1,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        >
          <Heart className="w-16 h-16 text-red-500 fill-red-500" />
        </motion.div>
        
        <motion.div
          className="absolute inset-0 rounded-full"
          animate={{
            scale: [1, 1.5, 1],
            opacity: [0.8, 0, 0.8],
          }}
          transition={{
            duration: 1,
            repeat: Infinity,
            ease: "easeInOut",
          }}
          style={{
            background: "radial-gradient(circle, rgba(239,68,68,0.3) 0%, rgba(239,68,68,0) 70%)",
          }}
        />
      </div>

      <div className="w-64 h-16 relative overflow-hidden bg-background border border-border rounded-lg p-4">
        <svg
          width="100%"
          height="100%"
          viewBox="0 0 256 64"
          className="absolute inset-0"
        >
          <motion.path
            d="M0,32 L60,32 L70,10 L80,54 L90,32 L100,32 L110,20 L120,44 L130,32 L256,32"
            stroke="currentColor"
            strokeWidth="2"
            fill="none"
            className="text-red-500"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          />
        </svg>
        
        <motion.div
          className="absolute top-0 left-0 w-1 h-full bg-red-500 opacity-80"
          animate={{
            x: [0, 256, 0],
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
      </div>

      <div className="flex space-x-1">
        {[...Array(3)].map((_, i) => (
          <motion.div
            key={i}
            className="w-2 h-2 bg-red-500 rounded-full"
            animate={{
              scale: [1, 1.5, 1],
              opacity: [0.5, 1, 0.5],
            }}
            transition={{
              duration: 1,
              repeat: Infinity,
              delay: i * 0.2,
            }}
          />
        ))}
      </div>
    </div>
  );
};

const StethoscopeLoader = ({ className }: { className?: string }) => {
  return (
    <div className={cn("flex flex-col items-center space-y-6", className)}>
      <motion.div
        animate={{
          rotate: [0, 10, -10, 0],
        }}
        transition={{
          duration: 2,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      >
        <Logo size={64} />
      </motion.div>

      <div className="relative">
        {[...Array(4)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-8 h-8 border-2 border-blue-500 rounded-full"
            style={{
              left: "50%",
              top: "50%",
              marginLeft: "-16px",
              marginTop: "-16px",
            }}
            animate={{
              scale: [0, 1.5],
              opacity: [1, 0],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              delay: i * 0.5,
            }}
          />
        ))}
      </div>

      <div className="flex space-x-2">
        {[...Array(5)].map((_, i) => (
          <motion.div
            key={i}
            className="w-1 bg-blue-500 rounded-full"
            animate={{
              height: [8, 24, 8],
            }}
            transition={{
              duration: 1.2,
              repeat: Infinity,
              delay: i * 0.1,
            }}
          />
        ))}
      </div>
    </div>
  );
};

const PulseLoader = ({ className }: { className?: string }) => {
  return (
    <div className={cn("flex flex-col items-center space-y-6", className)}>
      <div className="relative">
        <Activity className="w-16 h-16 text-green-500" />
        
        <motion.div
          className="absolute inset-0"
          animate={{
            scale: [1, 1.3, 1],
            opacity: [1, 0.3, 1],
          }}
          transition={{
            duration: 1.5,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        >
          <Activity className="w-16 h-16 text-green-400" />
        </motion.div>
      </div>

      <div className="w-48 h-12 relative bg-background border border-border rounded-lg overflow-hidden">
        <motion.div
          className="absolute inset-0 bg-gradient-to-r from-transparent via-green-500/20 to-transparent"
          animate={{
            x: [-48, 192],
          }}
          transition={{
            duration: 1.5,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
        
        <div className="absolute inset-0 flex items-center justify-center">
          <motion.div
            className="text-green-500 font-mono text-sm"
            animate={{
              opacity: [1, 0.5, 1],
            }}
            transition={{
              duration: 1,
              repeat: Infinity,
            }}
          >
            ••• ••• •••
          </motion.div>
        </div>
      </div>

      <motion.div
        className="text-sm text-muted-foreground"
        animate={{
          opacity: [0.5, 1, 0.5],
        }}
        transition={{
          duration: 2,
          repeat: Infinity,
        }}
      >
        Monitoring vitals...
      </motion.div>
    </div>
  );
};

export function HealthcareLoadingScreen({
  variant = "heartbeat",
  message = "Loading healthcare data...",
  className,
}: HealthcareLoadingScreenProps) {
  const renderLoader = () => {
    switch (variant) {
      case "heartbeat":
        return <HeartbeatLoader />;
      case "stethoscope":
        return <StethoscopeLoader />;
      case "pulse":
        return <PulseLoader />;
      default:
        return <HeartbeatLoader />;
    }
  };

  return (
    <div className={cn(
      "min-h-screen bg-background flex flex-col items-center justify-center p-8",
      className
    )}>
      <div className="flex flex-col items-center space-y-8">
        {renderLoader()}
        
        <motion.div
          className="text-center space-y-2"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
              <div className="flex items-center justify-center space-x-3">
                <Logo size={64} />
                <h2 className="text-2xl font-semibold text-foreground">CortexMD</h2>
              </div>
          <motion.p
            className="text-muted-foreground"
            animate={{
              opacity: [0.7, 1, 0.7],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
            }}
          >
            {message}
          </motion.p>
        </motion.div>

        <div className="flex space-x-1">
          {[...Array(3)].map((_, i) => (
            <motion.div
              key={i}
              className="w-2 h-2 bg-primary rounded-full"
              animate={{
                y: [0, -8, 0],
              }}
              transition={{
                duration: 0.8,
                repeat: Infinity,
                delay: i * 0.1,
              }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

export default function HealthcareLoadingScreenDemo() {
  return <HealthcareLoadingScreen variant="heartbeat" message="Connecting to patient database..." />;
}
