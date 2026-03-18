import fs from 'fs';

const users = 32;
const movies = 32;
const genres = 8;
const entries = 1000;

let csv = "user_id,movie_id,genre_id,rating\n";
const seen = new Set();

for (let i = 0; i < entries; i++) {
    let u, m, g;
    do {
        u = Math.floor(Math.random() * users);
        m = Math.floor(Math.random() * movies);
        g = Math.floor(Math.random() * genres);
    } while (seen.has(`${u}-${m}-${g}`));
    
    seen.add(`${u}-${m}-${g}`);
    
    // Some logic for "realistic" ratings:
    // User u likes genre g more
    const bias = (u % genres === g) ? 1.5 : 0;
    const rating = Math.min(5, Math.max(1, Math.round(Math.random() * 3 + 1 + bias)));
    
    csv += `${u},${m},${g},${rating}\n`;
}

fs.writeFileSync('c:/Users/rolan/OneDrive/Desktop/hypercube/showcase/assets/sample-tensor.csv', csv);
console.log("Generated sample-tensor.csv with 1000 entries.");
