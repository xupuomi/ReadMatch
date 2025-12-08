import React, { useEffect, useState } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Link,
  useNavigate,
  useParams,
  useLocation,
} from "react-router-dom";
import LoginPage from "./loginpage";

const API_BASE = "http://127.0.0.1:5000";

function displayName(username) {
  if (!username) return "Reader";
  const base = username.split("@")[0] || username;
  return base.charAt(0).toUpperCase() + base.slice(1);
}

export default function App() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [lastQuery, setLastQuery] = useState("");
  const [user, setUser] = useState(null); // { id, username }
  const [authReady, setAuthReady] = useState(false);

  useEffect(() => {
    const saved = localStorage.getItem("readmatch_user");
    if (saved) {
      try {
        setUser(JSON.parse(saved));
      } catch {
        localStorage.removeItem("readmatch_user");
      }
    }
    setAuthReady(true);
  }, []);

  function handleAuth(u) {
    if (u) {
      localStorage.setItem("readmatch_user", JSON.stringify(u));
    } else {
      localStorage.removeItem("readmatch_user");
    }
    setUser(u);
  }

  async function runSearch(query) {
    setLoading(true);
    setError("");
    setResults([]);
    setLastQuery(query);
    try {
      const resp = await fetch(`${API_BASE}/search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      setResults(data.results || []);
    } catch (err) {
      setError("Search failed. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <Router>
      <div className="min-h-screen bg-slate-50 text-slate-900">
        <Header userId={user?.id} />
        <main className="container mx-auto p-4">
          <Routes>
            <Route path="/" element={<LoginPage onAuth={handleAuth} user={user} authReady={authReady} />} />
            <Route
              path="/search"
              element={<HomePage runSearch={runSearch} user={user} authReady={authReady} loading={loading} />}
            />
            <Route
              path="/results"
              element={
                <ResultsPage
                  results={results}
                  loading={loading}
                  error={error}
                  query={lastQuery}
                  runSearch={runSearch}
                  user={user}
                  authReady={authReady}
                />
              }
            />
            <Route path="/book/:id" element={<BookDetailPage user={user} userId={user?.id} authReady={authReady} />} />
            <Route path="/profile" element={<ProfilePage user={user} onAuth={handleAuth} authReady={authReady} />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

/* ------------------------------ Header ------------------------------ */

function Header({ userId }) {
  return (
    <header className="bg-white shadow-sm">
      <div className="container mx-auto flex items-center justify-between p-3">
        <Link to="/search" className="text-2xl font-bold">
          ReadMatch
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

function HomePage({ runSearch, user, authReady, loading }) {
  const navigate = useNavigate();
  const [query, setQuery] = useState("");

  useEffect(() => {
    if (!authReady) return;
    if (!user?.id) {
      navigate("/");
    }
  }, [user, authReady, navigate]);

  if (!authReady) return null;

  function onSearch(e) {
    e.preventDefault();
    const q = query.trim();
    if (!q) return;
    runSearch(q).then(() => navigate("/results"));
  }

  return (
    <div className="max-w-3xl mx-auto">
      <h1 className="text-3xl font-semibold mb-6">Find your next favorite book…</h1>

      <form onSubmit={onSearch}>
        <div className="flex gap-2">
          <input
            className="flex-1 border rounded p-2"
            placeholder="Search by title or genre"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />

          <button className="bg-sky-600 text-white px-4 py-2 rounded">
            Search
          </button>
        </div>
      </form>
      {loading && <div className="text-slate-500 mt-3">Searching…</div>}
    </div>
  );
}

function ResultsPage({ results, loading, error, query, runSearch, user, authReady }) {
  const [localQuery, setLocalQuery] = useState(query || "");
  const navigate = useNavigate();

  useEffect(() => {
    setLocalQuery(query || "");
  }, [query]);

  useEffect(() => {
    if (!authReady) return;
    if (!user?.id) {
      navigate("/");
    }
  }, [user, authReady, navigate]);

  if (!authReady) return null;

  async function search(e) {
    e.preventDefault();
    const q = localQuery.trim();
    if (!q) return;
    await runSearch(q);
  }

  return (
    <div className="max-w-3xl mx-auto">
      <h1 className="text-3xl font-semibold mb-4">Find your next favorite book…</h1>

      <form onSubmit={search} className="mt-2 mb-4 flex gap-2">
        <input
          className="flex-1 border rounded p-2 text-slate-900"
          placeholder="Search by title, author or genre"
          value={localQuery}
          onChange={(e) => setLocalQuery(e.target.value)}
        />
        <button className="bg-sky-600 text-white px-4 py-2 rounded">
          Search
        </button>
      </form>

      {loading && <div className="text-slate-500 mt-4">Searching…</div>}
      {error && <div className="text-red-600 mt-4">{error}</div>}

      {!loading && !error && (
        <div className="mt-6 space-y-3">
          {results.length === 0 ? (
            <div className="text-slate-500 mt-4">
              {query ? `No results for "${query}".` : "Results will appear here once you search."}
            </div>
          ) : (
            results.map((b) => (
              <button
                key={b.book_id}
                className="w-full text-left border rounded p-3 bg-white shadow-sm hover:border-sky-300 hover:shadow"
                onClick={() => navigate(`/book/${b.book_id}`, { state: { from: "/results", query: localQuery } })}
                type="button"
              >
                <div className="font-semibold">{b.title}</div>
                <div className="text-sm text-slate-600">{b.authors}</div>
                <div className="text-sm text-slate-500">{b.genres}</div>
                <StarDisplay rating={b.avg_rating} count={b.review_count} />
                <div className="text-sm text-slate-700 mt-1 line-clamp-3">
                  {b.description}
                </div>
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}

function BookDetailPage({ user, userId, authReady }) {
  const { id } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const [book, setBook] = useState(null);
  const [statusMsg, setStatusMsg] = useState("");
  const [rating, setRating] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!authReady) return;
    if (!user?.id) {
      navigate("/");
      return;
    }
    let alive = true;
    async function load() {
      setLoading(true);
      setError("");
      try {
        const url = userId ? `${API_BASE}/book/${id}?user_id=${userId}` : `${API_BASE}/book/${id}`;
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        if (alive) {
          setBook(data);
          if (data.user_rating) setRating(Number(data.user_rating));
        }
      } catch (err) {
        if (alive) setError("Failed to load book.");
      } finally {
        if (alive) setLoading(false);
      }
    }
    load();
    return () => {
      alive = false;
    };
  }, [id]);

  return (
    <div className="max-w-3xl mx-auto bg-white p-6 rounded shadow">
      {loading && <div className="text-slate-500">Loading…</div>}
      {error && <div className="text-red-600">{error}</div>}
      {!loading && !error && book && (
        <div className="flex gap-6">
          {/* Placeholder book cover */}
          <div className="w-40 h-56 bg-gray-200 rounded" />

          <div className="flex-1">
            <h1 className="text-2xl font-bold">{book.title}</h1>
            <div className="text-sm text-slate-600">{book.authors}</div>
            <div className="mt-2">Genre: {book.genres}</div>
            <StarDisplay rating={book.avg_rating} count={book.review_count} />

            <div className="mt-4">
              <h3 className="font-semibold">Description</h3>
              <p className="mt-1 text-slate-700">{book.description}</p>
            </div>

            <div className="mt-6 flex gap-3 flex-wrap">
              <button
                className="px-3 py-1 border rounded text-sm"
                type="button"
                onClick={() => {
                  const backQuery = location.state?.query;
                  const backTo = location.state?.from || "/results";
                  navigate(backTo, backQuery ? { state: { query: backQuery } } : undefined);
                }}
              >
                ← Back to results
              </button>
              <button
                className="px-4 py-2 bg-sky-600 text-white rounded"
                onClick={() => addWantToRead(id, setStatusMsg, userId)}
                type="button"
              >
                Save to Want to Read
              </button>

              <StarPicker
                value={rating}
                onChange={(val) => setRating(val)}
                onSubmit={() => submitRating(id, rating, setStatusMsg, userId)}
              />

              <button
                type="button"
                className="px-3 py-1 border rounded text-center"
                onClick={() => {
                  if (window.history.length > 1) {
                    navigate(-1);
                  } else {
                    navigate("/results");
                  }
                }}
              >
                Close
              </button>
            </div>
            {statusMsg && <div className="text-sm text-slate-600 mt-2">{statusMsg}</div>}
          </div>
        </div>
      )}
    </div>
  );
}

function ProfilePage({ user, onAuth, authReady }) {
  const [saved, setSaved] = useState([]);
  const [ratings, setRatings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    let alive = true;
    async function load() {
      if (!authReady) return;
      if (!user?.id) {
        setSaved([]);
        setRatings([]);
        setLoading(false);
        navigate("/");
        return;
      }
      if (!user?.id) {
        setSaved([]);
        setRatings([]);
        setLoading(false);
        return;
      }
      setLoading(true);
      setError("");
      try {
        const [savedResp, ratingsResp] = await Promise.all([
          fetch(`${API_BASE}/users/${user.id}/want_to_read`),
          fetch(`${API_BASE}/users/${user.id}/ratings`),
        ]);
        if (!savedResp.ok || !ratingsResp.ok) throw new Error("HTTP error");
        const savedData = await savedResp.json();
        const ratingsData = await ratingsResp.json();
        if (alive) {
          setSaved(savedData.books || []);
          setRatings(ratingsData.ratings || []);
        }
      } catch (err) {
        if (alive) setError("Failed to load profile data.");
      } finally {
        if (alive) setLoading(false);
      }
    }
    load();
    return () => {
      alive = false;
    };
  }, [user, authReady, navigate]);

  if (!authReady) return null;

  return (
    <div className="max-w-3xl mx-auto bg-white p-6 rounded shadow">
      <h1 className="text-2xl font-semibold">User Profile</h1>

      {!user?.id && <div className="text-slate-600 mt-4">Please log in to view your saved books and reviews.</div>}

      {user?.id && (
        <div className="mt-2">
          <div className="text-lg font-semibold">Hello, {displayName(user?.username)}!</div>
          <button
            type="button"
            className="mt-2 px-3 py-1 border rounded text-sm text-slate-700 hover:bg-slate-50"
            onClick={() => {
              onAuth && onAuth(null);
              navigate("/");
            }}
          >
            Sign out
          </button>
        </div>
      )}

      {loading && <div className="text-slate-500 mt-4">Loading…</div>}
      {error && <div className="text-red-600 mt-4">{error}</div>}

      {!loading && !error && (
        <>
          <div className="mt-4">
            <h3 className="font-semibold">Saved Books</h3>
            {saved.length === 0 ? (
              <p className="text-slate-600 text-sm">No saved books yet.</p>
            ) : (
              <div className="mt-2 space-y-2">
                {saved.map((b) => (
                  <div
                    key={b.book_id}
                    className="border rounded bg-white shadow-sm hover:border-sky-300 hover:shadow flex"
                  >
                    <Link to={`/book/${b.book_id}`} className="flex-1 p-3 block">
                      <div className="font-semibold">{b.title}</div>
                      <div className="text-sm text-slate-600">{b.authors}</div>
                      <StarDisplay rating={b.avg_rating} count={b.review_count} />
                    </Link>
                    <div className="border-l px-3 py-2 flex items-center">
                      <button
                        type="button"
                        className="px-3 py-1 text-sm border rounded bg-slate-50 hover:bg-slate-100"
                        onClick={(e) => {
                          e.preventDefault();
                          e.stopPropagation();
                          removeSavedBook(b.book_id, setSaved, setError, user?.id);
                        }}
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="mt-6">
            <h3 className="font-semibold">Past Reviews</h3>
            {ratings.length === 0 ? (
              <p className="text-slate-600">No reviews yet.</p>
            ) : (
              <div className="mt-2 space-y-2">
                {ratings.map((r) => (
                  <div
                    key={r.book_id}
                    className="border rounded bg-white shadow-sm hover:border-sky-300 hover:shadow flex"
                  >
                    <Link to={`/book/${r.book_id}`} className="flex-1 p-3 block">
                      <div className="flex items-center gap-3">
                        <div>
                          <div className="font-semibold">{r.title}</div>
                          <div className="text-sm text-slate-600">{r.authors}</div>
                        </div>
                        <div className="flex items-center gap-1 text-lg" title="Your rating">
                          {[1, 2, 3, 4, 5].map((i) => (
                            <span key={i} className={i <= Number(r.rating) ? "text-amber-400" : "text-slate-300"}>
                              ★
                            </span>
                          ))}
                          <span className="text-sm text-slate-500">(your rating)</span>
                        </div>
                      </div>
                    </Link>
                    <div className="border-l px-3 py-2 flex items-center">
                      <button
                        type="button"
                        className="px-3 py-1 text-sm border rounded bg-slate-50 hover:bg-slate-100"
                        onClick={(e) => {
                          e.preventDefault();
                          e.stopPropagation();
                          removeRating(r.book_id, setRatings, setError, user?.id);
                        }}
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

async function addWantToRead(bookId, setStatus, userId) {
  if (!userId) {
    setStatus && setStatus("Please log in first.");
    return;
  }
  try {
    const resp = await fetch(`${API_BASE}/users/${userId}/want_to_read`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ book_id: Number(bookId) }),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    setStatus && setStatus("Saved to Want to Read.");
  } catch (err) {
    setStatus && setStatus("Failed to save. Try again.");
  }
}

async function submitRating(bookId, rating, setStatus, userId) {
  if (!userId) {
    setStatus && setStatus("Please log in first.");
    return;
  }
  try {
    const resp = await fetch(`${API_BASE}/users/${userId}/rating`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ book_id: Number(bookId), rating }),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    setStatus && setStatus("Rating saved.");
  } catch (err) {
    setStatus && setStatus("Failed to rate. Try again.");
  }
}

function StarDisplay({ rating, count }) {
  const r = Number(rating) || 0;
  const filled = Math.floor(r);
  const hasHalf = r - filled >= 0.5 && filled < 5;

  const halfStarStyle = {
    background: "linear-gradient(90deg, #fbbf24 50%, #e5e7eb 50%)",
    WebkitBackgroundClip: "text",
    WebkitTextFillColor: "transparent",
  };

  return (
    <div className="flex items-center gap-2 text-sm text-slate-600">
      <div className="flex">
        {[1, 2, 3, 4, 5].map((i) => (
          <span
            key={i}
            className={
              i <= filled
                ? "text-amber-400"
                : hasHalf && i === filled + 1
                ? ""
                : "text-slate-300"
            }
            style={hasHalf && i === filled + 1 ? halfStarStyle : undefined}
          >
            ★
          </span>
        ))}
      </div>
      <span>
        {rating ?? "—"} ({count ?? 0})
      </span>
    </div>
  );
}

function StarPicker({ value, onChange, onSubmit, compact }) {
  return (
    <div className={`flex items-center gap-2 ${compact ? "text-sm" : ""}`}>
      <div className="flex">
        {[1, 2, 3, 4, 5].map((i) => (
          <button
            key={i}
            type="button"
            className="px-1 text-xl"
            onClick={() => onChange(i)}
          >
            <span className={i <= value ? "text-amber-400" : "text-slate-300"}>★</span>
          </button>
        ))}
      </div>
      <button
        type="button"
        className="px-2 py-1 border rounded text-sm"
        onClick={onSubmit || (() => {})}
      >
        Save Rating
      </button>
    </div>
  );
}

async function removeSavedBook(bookId, setSaved, setError, userId) {
  if (!userId) {
    setError && setError("Please log in first.");
    return;
  }
  try {
    const resp = await fetch(`${API_BASE}/users/${userId}/want_to_read/${bookId}`, {
      method: "DELETE",
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    setSaved((prev) => prev.filter((b) => b.book_id !== bookId));
  } catch (err) {
    setError && setError("Failed to remove saved book.");
  }
}

async function removeRating(bookId, setRatings, setError, userId) {
  if (!userId) {
    setError && setError("Please log in first.");
    return;
  }
  try {
    const resp = await fetch(`${API_BASE}/users/${userId}/rating/${bookId}`, {
      method: "DELETE",
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    setRatings((prev) => prev.filter((r) => r.book_id !== bookId));
  } catch (err) {
    setError && setError("Failed to remove rating.");
  }
}
