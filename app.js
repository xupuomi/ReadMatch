import React, { useState } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Link,
  useNavigate,
  useParams,
} from "react-router-dom";

export default function App() {
  return (
    <Router>
      <div className="min-h-screen bg-slate-50 text-slate-900">
        <Header />
        <main className="container mx-auto p-4">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/results" element={<ResultsPage />} />
            <Route path="/book/:id" element={<BookDetailPage />} />
            <Route path="/profile" element={<ProfilePage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

/* ------------------------------ Header ------------------------------ */

function Header() {
  return (
    <header className="bg-white shadow-sm">
      <div className="container mx-auto flex items-center justify-between p-3">
        <Link to="/" className="text-2xl font-bold">
          BookFinder
        </Link>

        <nav className="flex gap-4">
          <Link to="/profile" className="hover:underline">
            Profile
          </Link>
        </nav>
      </div>
    </header>
  );
}

/* ------------------------------ Home Page ------------------------------ */

function HomePage() {
  const navigate = useNavigate();
  const [query, setQuery] = useState("");

  function onSearch(e) {
    e.preventDefault();
    navigate("/results"); // backend will handle actual results later
  }

  return (
    <div className="max-w-3xl mx-auto">
      <h1 className="text-3xl font-semibold mb-6">Find your next favorite book…</h1>

      <form onSubmit={onSearch}>
        <div className="flex gap-2">
          <input
            className="flex-1 border rounded p-2"
            placeholder="Search by title, author or genre"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />

          <button className="bg-sky-600 text-white px-4 py-2 rounded">
            Search
          </button>
        </div>
      </form>
    </div>
  );
}

/* ------------------------------ Results Page ------------------------------ */

function ResultsPage() {
  return (
    <div>
      <h2 className="text-2xl font-semibold mb-4">Recommended Books</h2>

      {/* Placeholder grid for teammates to render real data */}
      <div className="text-slate-500 mt-10 text-center">
        Results will appear here once backend is connected.
      </div>
    </div>
  );
}

/* ------------------------------ Book Detail Page ------------------------------ */

function BookDetailPage() {
  const { id } = useParams();

  return (
    <div className="max-w-3xl mx-auto bg-white p-6 rounded shadow">
      <div className="flex gap-6">
        {/* Placeholder book cover */}
        <div className="w-40 h-56 bg-gray-200 rounded" />

        <div className="flex-1">
          <h1 className="text-2xl font-bold">Book Title</h1>
          <div className="text-sm text-slate-600">Author Name</div>
          <div className="mt-2">Genre: —</div>
          <div className="mt-1">Rating: — ⭐</div>

          <div className="mt-4">
            <h3 className="font-semibold">Description</h3>
            <p className="mt-1 text-slate-700">Description goes here…</p>
          </div>

          <div className="mt-6 flex gap-3">
            <button className="px-4 py-2 bg-sky-600 text-white rounded">
              Save to Want to Read
            </button>

            <div className="flex items-center gap-1">
              {[1, 2, 3, 4, 5].map((s) => (
                <button key={s} className="px-2 py-1 border rounded bg-white">
                  ⭐
                </button>
              ))}
            </div>

            <Link to="/results" className="px-3 py-1 border rounded text-center">
              Close
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------ Profile Page ------------------------------ */

function ProfilePage() {
  return (
    <div className="max-w-3xl mx-auto bg-white p-6 rounded shadow">
      <h1 className="text-2xl font-semibold">User Profile</h1>

      <div className="mt-4">
        <h3 className="font-semibold">Saved Books</h3>
        <p className="text-slate-600 text-sm">
          Your saved items will appear here once connected.
        </p>
      </div>

      <div className="mt-6">
        <h3 className="font-semibold">Past Reviews</h3>
        <p className="text-slate-600">No reviews yet.</p>
      </div>
    </div>
  );
}
