import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

const API_BASE = "http://127.0.0.1:5000";

export default function LoginPage({ onAuth, user }) {
	const navigate = useNavigate();
	const [email, setEmail] = useState("");
	const [password, setPassword] = useState("");
	const [error, setError] = useState("");

	useEffect(() => {
		if (user && user.id) {
			navigate("/search");
		}
	}, [user, navigate]);

  async function onLogin(e) {
    e.preventDefault();
    setError("");
    const username = email.trim();
    if (!username || !password.trim()) return;

    // try login
    const loginResp = await fetch(`${API_BASE}/users/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });

    if (loginResp.ok) {
      const data = await loginResp.json();
      onAuth && onAuth({ id: data.user_id, username });
      navigate("/search");
      return;
    }

    // if login failed, try register
    const registerResp = await fetch(`${API_BASE}/users/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });

    if (registerResp.ok) {
      const data = await registerResp.json();
      onAuth && onAuth({ id: data.user_id, username });
      navigate("/search");
      return;
    }

    if (registerResp.status === 409) {
      setError("Incorrect password for this email.");
      return;
    }

    setError("Login/Register failed. Check credentials.");
  }

  function onLogout() {
    onAuth && onAuth(null);
    setEmail("");
    setPassword("");
    navigate("/");
  }

  return (
    <div className="max-w-md mx-auto bg-white p-8 rounded-lg shadow-xl mt-10">
      <h1 className="text-3xl font-bold mb-6 text-center">Welcome to BookFinder</h1>
      <p className="text-center text-slate-600 mb-6">
        Log in or create a profile to find your next book!
      </p>

      <form onSubmit={onLogin} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-slate-700">Email</label>
          <input
            type="email"
            required
            className="w-full border rounded p-2 mt-1"
            placeholder="Enter your email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-slate-700">Password</label>
          <input
            type="password"
            required
            className="w-full border rounded p-2 mt-1"
            placeholder="Enter your password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </div>
        <button
          type="submit"
          className="w-full bg-sky-600 text-white font-semibold px-4 py-2 rounded hover:bg-sky-700 transition duration-150"
        >
          Login / Create Profile
        </button>
        {user && user.id && (
          <button
            type="button"
            className="w-full border border-sky-200 text-sky-700 font-semibold px-4 py-2 rounded hover:bg-sky-50 transition duration-150"
            onClick={onLogout}
          >
            Log out
          </button>
        )}
        {error && <div className="text-red-600 text-sm">{error}</div>}
      </form>
    </div>
  );
}
