import React from "react";
import { Link } from "react-router-dom";
import { useLocation } from "react-router-dom";
import { useNavigate } from "react-router-dom";

export default function UserProfile() {
  const location = useLocation();
  const userData = location.state?.userData;
  console.log("User Data:", userData);
  let username = userData.username;
  const parsedJsonData = JSON.parse(userData.datasets);
  const navigate = useNavigate();
  //make the datasetID in the json object to id in another json the name is content the date 2023-01-01
  const newjson = parsedJsonData.map((item, index) => {
    return {
      id: item.datasetID,
      content: item.datasetName,
      date: "2023-01-01",
    };
  });
  // console.log("New JSON:", newjson);
  // console.log("User Image:", userData.image);
  const userimageurl = `data:image/jpeg;base64,${userData.userimage}`;
  const user = {
    name: username,
    bio: "A data science student who enjoys learning new stuff.",
    profilePicture: userimageurl,
    posts: newjson,
  };
  const sendDatasetToServer = async (postid) => {
    const formData = new FormData();
    formData.append("postid", postid);
    try {
      const response = await fetch(
        "http://127.0.0.1:8000/retTuner/getsavedmodel",
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const result = await response.json();
      if (result.status === "success") {
        const modelData = {
          hyperparameters: result.hyperparameters,
        };
        navigate("/Output", { state: { modelData } });
      }
      //     /
    } catch (error) {
      console.error("Error sending dataset to server:", error);
    }
  };
  const handleClick = (postid) => {
    console.log("Post ID:", postid);
    sendDatasetToServer(postid);
  };

  //   [
  //     { id: 1, content: "This is my first project!", date: "2023-01-01" },
  //     { id: 2, content: "Hello!", date: "2023-02-01" },
  //     { id: 3, content: "Another project here.", date: "2023-03-01" },
  //   ],

  return (
    <div id="profile" className="py-5">
      <div className="container w-75 py-5">
        <div className="row">
          <div className="col-md-4 text-center">
            <img
              src={user.profilePicture}
              alt="Profile"
              className="rounded-circle shadow-sm"
              style={{ width: "150px", height: "150px" }}
            />
            <h2 className="mt-3">{user.name}</h2>
            <p>{user.bio}</p>
            <button className="btn btn-danger mt-3">Edit Profile</button>
          </div>
          <div className="col-md-8">
            <div className="posts bg-white p-4 rounded shadow-sm">
              <h3 className="mb-4">Projects</h3>
              <div className="grid-container">
                {user.posts.map((post) => (
                  <div key={post.id} className="post mb-3">
                    <div className="post-content bg-light p-3 rounded">
                      <p className="post-date">
                        <strong>Date:</strong> {post.date}
                      </p>
                      <p>{post.content}</p>
                      <div className="col-md-12 text-center">
                        <button
                          className="btn btn-outline-secondary"
                          style={{ backgroundColor: "#dc3545", color: "white" }}
                          onClick={() => handleClick(post.id)}
                        >
                          Use Model
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
        <div className="row mt-5">
          <div className="col-md-12 text-center">
            <Link
              to="/"
              className="btn btn-outline-secondary"
              style={{
                backgroundColor: "#dc3545",
                color: "white",
                marginRight: "10px",
              }}
            >
              Back to Home
            </Link>
            <Link
              to="/getmodel"
              className="btn btn-outline-secondary"
              style={{
                backgroundColor: "#dc3545",
                color: "white",
                marginRight: "10px",
              }}
            >
              Create New Project
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
