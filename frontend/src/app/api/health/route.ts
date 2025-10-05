import { NextResponse } from 'next/server';

export async function GET() {
  try {
    // Basic health check
    const health = {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      service: 'cortexmd-frontend',
      version: '1.0.0',
      uptime: process.uptime(),
      memory: process.memoryUsage(),
    };

    return NextResponse.json(health, { status: 200 });
  } catch (error) {
    const errorHealth = {
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      service: 'cortexmd-frontend',
      error: error instanceof Error ? error.message : 'Unknown error',
    };

    return NextResponse.json(errorHealth, { status: 503 });
  }
}