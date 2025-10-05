'use client';

export function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-slate-800 text-white py-8 mt-auto">
      <div className="container mx-auto px-4">
        <div className="grid md:grid-cols-4 gap-8">
          {/* Brand */}
          <div className="md:col-span-2">
            <div className="text-2xl font-bold mb-4 text-gradient">
              🧠 CortexMD
            </div>
            <p className="text-slate-400 max-w-md">
              Advanced AI-powered medical diagnosis system with FOL verification 
              and explainable AI insights for healthcare professionals.
            </p>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className="text-lg font-semibold mb-4">Quick Links</h3>
            <ul className="space-y-2">
              <li>
                <a href="/" className="text-slate-400 hover:text-white transition-colors">
                  Diagnosis
                </a>
              </li>
              <li>
                <a href="/chat" className="text-slate-400 hover:text-white transition-colors">
                  AI Chat
                </a>
              </li>
              <li>
                <a href="/umls" className="text-slate-400 hover:text-white transition-colors">
                  UMLS Lookup
                </a>
              </li>
              <li>
                <a href="/about" className="text-slate-400 hover:text-white transition-colors">
                  About
                </a>
              </li>
            </ul>
          </div>

          {/* Contact */}
          <div>
            <h3 className="text-lg font-semibold mb-4">Features</h3>
            <ul className="space-y-2 text-slate-400">
              <li>• FOL Verification</li>
              <li>• Medical Imaging</li>
              <li>• FHIR Integration</li>
              <li>• Real-time Analysis</li>
              <li>• Knowledge Graph</li>
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="border-t border-slate-700 mt-8 pt-8 flex flex-col md:flex-row justify-between items-center">
          <p className="text-slate-400 text-sm">
            © {currentYear} CortexMD. All rights reserved.
          </p>
          <div className="flex space-x-4 mt-4 md:mt-0">
            <span className="text-slate-400 text-sm">
              Built for Samsung Gen AI Hackathon
            </span>
          </div>
        </div>
      </div>
    </footer>
  );
}
