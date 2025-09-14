import React from "react";

function Recommendations({ list }) {
  return (
    <div className="recommendations">
      <ul>
        {list.map((book, idx) => (
          <li key={idx}>{book}</li>
        ))}
      </ul>
    </div>
  );
}

export default Recommendations;
