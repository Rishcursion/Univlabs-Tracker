
import React from 'react';

export const Header = () => {
  return (
    <header className="bg-slate-800 border-b border-slate-700">
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-lg">U</span>
            </div>
            <div>
              <h1 className="text-xl font-semibold text-white">Surgical Vision Scribe</h1>
              <p className="text-slate-400 text-sm">Advanced Video Annotation Platform</p>
            </div>
          </div>
          <div className="text-slate-400 text-sm">
            Powered by Univlabs
          </div>
        </div>
      </div>
    </header>
  );
};
