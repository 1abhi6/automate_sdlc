"use client";

import { useState, useEffect } from "react";

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [greeting, setGreeting] = useState("");

  // Dynamically set greeting
  useEffect(() => {
    const hour = new Date().getHours();
    if (hour < 12) {
      setGreeting("Good Morning, User ☀️");
    } else if (hour < 18) {
      setGreeting("Good Afternoon, User 🌤️");
    } else {
      setGreeting("Good Evening, User 🌙");
    }
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    alert("User asked: " + prompt); // replace with API logic later
    setPrompt("");
  };

  return (
    <div className="flex flex-col justify-between min-h-screen bg-gradient-to-r from-indigo-200 via-purple-200 to-pink-200">
      {/* Greeting at top */}
      <header className="p-20 text-center">
        <h1 className="text-7xl font-extrabold text-gray-900 drop-shadow-xl animate-pulse">
            {greeting}
        </h1>
      </header>


      {/* Spacer for middle content (optional chat history later) */}
    <main className="flex-1 flex items-center justify-center">
      <p className="text-3xl font-semibold text-gray-800 opacity-80">
        Start your conversation below 👇
      </p>
    </main>


      {/* Chat input fixed at bottom */}
        <footer className="sticky bottom-0 w-full p-4 bg-gradient-to-r from-indigo-200 via-purple-200 to-pink-200 backdrop-blur-sm border-t border-gray-300">
        <form
          onSubmit={handleSubmit}
          className="w-full max-w-2xl mx-auto flex items-center gap-3 bg-white shadow-md rounded-2xl p-3"
        >
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Ask me anything..."
            className="flex-1 p-3 rounded-xl outline-none bg-gray-50 text-gray-800 border border-gray-300 focus:ring-2 focus:ring-purple-400 transition"
          />
          <button
            type="submit"
            className="px-5 py-2 bg-purple-600 text-white rounded-xl font-medium hover:bg-purple-700 transition"
          >
            Send
          </button>
        </form>
      </footer>
    </div>
  );
}



