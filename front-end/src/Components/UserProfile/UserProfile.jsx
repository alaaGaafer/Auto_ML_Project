import React, { useState } from "react";
import { Link } from "react-router-dom";
import { useLocation } from "react-router-dom";
import { useNavigate } from "react-router-dom";
import { useContext } from "react";
import { AuthContextUser } from "../../Context/Contextt";
import { dataContext } from "../../Context/Context";

export default function UserProfile() {
  const { ShareFile, setShareFile } = useContext(dataContext);
  const { user, setUser } = useContext(AuthContextUser);
  const [phone, setPhone] = useState(user ? user.phone : "");
  let userData = user;
  let username = userData.username;
  let parsedJsonData = JSON.parse(userData.datasets);
  const navigate = useNavigate();
  // console.log("the user data: ", userData);
  const updateDatasetId = (newDatasetId) => {
    setShareFile((prevState) => ({
      ...prevState,
      datasetid: newDatasetId,
    }));
  };
  const newjson = parsedJsonData.map((item, index) => {
    return {
      id: item.datasetID,
      content: item.datasetName,
      date: item.date,
      description: item.description,
      problemtype: item.problemType,
      accuracy: item.modelaccuracy,
      MSE: item.modelmse,
      modelname: item.modelname,
    };
  });

  const userimageurl = `data:image/jpeg;base64,${userData.userimage}`;
  const userr = {
    name: username,
    bio: "A data science student who enjoys learning new stuff.",
    profilePicture: userimageurl,
    posts: newjson,
  };

  const handleClick = (post) => {
    console.log("Post :", post);
    const modelData = {
      accuracy: post.accuracy,
      MSE: post.MSE,
      modelname: post.modelname,
    };
    console.log("Model Data:", modelData);
    updateDatasetId(post.id);
    navigate("/output", { state: { modelData } });
    // sendDatasetToServer(postid);
  };
  return (
    <div id="profile" className="py-5">
      <div className="container w-75 py-5">
        <div className="row">
          <div className="col-md-4 text-center">
            <img
              src={userr.profilePicture}
              alt="Profile"
              className="rounded-circle shadow-sm"
              style={{ width: "150px", height: "150px" }}
            />
            <h2 className="mt-3">{userr.name}</h2>
            <p>{userr.bio}</p>
            <button className="btn btn-danger mt-3">Edit Profile</button>
          </div>
          <div className="col-md-8">
            <div className="posts bg-white p-4 rounded shadow-sm">
              <h3 className="mb-4">Projects</h3>
              <div className="grid-container">
                {userr.posts.map((post) => (
                  <div key={post.id} className="post mb-3">
                    <div className="post-content bg-light p-3 rounded">
                      <p className="post-date">
                        <strong>Dataset ID:</strong> {post.id} <br />
                        <strong>Dataset Name:</strong> {post.content} <br />
                        <strong>Date:</strong> {post.date}
                        <br />
                        <strong>Problem Type:</strong> {post.problemtype} <br />
                        <strong>Description: </strong>
                        {post.description}
                      </p>
                      {/* <p>{post.content}</p> */}
                      {/* <p> description: {post.description}</p> */}
                      <div className="col-md-12 text-center">
                        <button
                          className="btn btn-outline-secondary"
                          style={{ backgroundColor: "#dc3545", color: "white" }}
                          onClick={() => handleClick(post)}
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
