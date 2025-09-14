import React, { useState } from "react";
import SearchBar from "./components/SearchBar";
import Recommendations from "./components/Recommendations";
import "./App.css"; // add CSS for split layout

function App() {
  const [bookRecs, setBookRecs] = useState([]);
  const [themeRecs, setThemeRecs] = useState([]);

  const fetchRecommendations = async (query, mode) => {
    const response = await fetch("http://127.0.0.1:5000/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, mode }), // mode = "book" or "theme"
    });
    const data = await response.json();

    if (mode === "book") {
      setBookRecs(data.recommendations);
    } else {
      setThemeRecs(data.recommendations);
    }
  };

  return (
    <div className="app-container">
      {/* Left side: Book similarity */}
      <div className="half left">
        <h2>📖 Find Similar Books</h2>
        <SearchBar
          placeholder="Enter a book title..."
          onSearch={(query) => fetchRecommendations(query, "book")}
        />
        <Recommendations list={bookRecs} />
      </div>

      {/* Right side: Theme search */}
      <div className="half right">
        <h2>🌌 Find Books by Theme</h2>
        <SearchBar
          placeholder="Enter a theme/topic..."
          onSearch={(query) => fetchRecommendations(query, "theme")}
        />
        <Recommendations list={themeRecs} />
      </div>
    </div>
  );
}

export default App;
