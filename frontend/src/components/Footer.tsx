
import React from 'react';

export const Footer = () => {
	return (
		<footer className="bg-slate-800 border-t border-slate-700 mt-16">
			<div className="container mx-auto px-6 py-6">
				<div className="flex items-center justify-center space-x-2 text-slate-400 text-sm">
					<span>© {new Date().getFullYear()} Univlabs</span>
					<span>•</span>
					<span>Surgical Video Analysis Platform</span>
				</div>
			</div>
		</footer>
	);
};
