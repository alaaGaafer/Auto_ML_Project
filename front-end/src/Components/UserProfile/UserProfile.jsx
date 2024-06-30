import React from 'react';
import { Link } from 'react-router-dom';

export default function UserProfile() {
    const user = {
        name: "Ahmed Ahmed",
        bio: "A data science student who enjoys learning new stuff.",
        profilePicture: "https://via.placeholder.com/150",
        posts: [
            { id: 1, content: "This is my first project!", date: "2023-01-01" },
            { id: 2, content: "Hello!", date: "2023-02-01" },
            { id: 3, content: "Another project here.", date: "2023-03-01" },
        ],
    };

    return (
        <div id='profile' className='py-5'>
            <div className="container w-75 py-5">
                <div className="row">
                    <div className="col-md-4 text-center">
                        <img src={user.profilePicture} alt="Profile" className="rounded-circle shadow-sm" style={{ width: '150px', height: '150px' }} />
                        <h2 className="mt-3">{user.name}</h2>
                        <p>{user.bio}</p>
                        <button className='btn btn-danger mt-3'>Edit Profile</button>
                    </div>
                    <div className="col-md-8">
                        <div className="posts bg-white p-4 rounded shadow-sm">
                            <h3 className="mb-4">Projects</h3>
                            <div className="grid-container">
                                {user.posts.map(post => (
                                    <div key={post.id} className="post mb-3">
                                        <div className="post-content bg-light p-3 rounded">
                                            <p className="post-date"><strong>Date:</strong> {post.date}</p>
                                            <p>{post.content}</p>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
                <div className="row mt-5">
                    <div className="col-md-12 text-center">
                        <Link to="/" className="btn btn-outline-secondary">Back to Home</Link>
                    </div>
                </div>
            </div>
        </div>
    );
}

