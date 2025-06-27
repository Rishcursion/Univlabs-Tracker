## MedSAM2 Interface

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.0%2B-61DAFB.svg)](https://reactjs.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Proof%20of%20Concept-orange.svg)]()

## Overview

MedSAM2 Interface is a cutting-edge medical image segmentation application designed as part of the **Collaborative Surgery Module** for UnivLabs. This proof-of-concept platform enables medical professionals and researchers to upload surgical videos and perform advanced segmentation tasks with precision and ease.

### Video Demo
<video width="360" height="240" controls >
  <source src="./assets/output.mp4" type="video/mp4">
</video>
### Key Features

- **Surgical Video Upload**: Seamlessly upload and process surgical video files
- **Interactive Annotation**: Annotate regions of interest with custom labels for tracking
- **Instance Segmentation**: Perform automatic segmentation of anatomical structures
- **Real-time Processing**: Leverage MedSAM2's advanced AI capabilities for medical imaging
- **Collaborative Workflow**: Designed for multi-user surgical training and analysis
- **Intuitive Interface**: User-friendly web interface for medical professionals

### Use Cases

- **Surgical Training**: Annotate key anatomical structures for educational purposes
- **Research Analysis**: Track and analyze surgical procedures for research studies
- **Quality Assessment**: Evaluate surgical techniques through detailed segmentation
- **Collaborative Learning**: Share and discuss surgical cases with team members

## Architecture

The application follows a modern full-stack architecture:

- **Frontend**: React-based web interface for user interactions
- **Backend**: FastAPI server handling AI processing and data management
- **AI Engine**: MedSAM2 model for medical image segmentation
- **Data Processing**: Efficient video processing and annotation pipeline

## Prerequisites

Before setting up the application, ensure you have the following installed:

- **Python 3.8+**
- **Node.js 16+**
- **npm or yarn**
- **CUDA-compatible GPU** (recommended for optimal performance)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Rishcursion/Univlabs-Traacker.git
cd medsam2-interface
```

### 2. Backend Setup

The backend installation and configuration has been completed. Please refer to the backend documentation for detailed setup instructions.

### 3. Frontend Setup

Navigate to the frontend directory and install dependencies:

```bash
cd frontend
npm install
```

## Running the Application

### Starting the Backend Server

Navigate to the backend directory and start the FastAPI server:

```bash
cd backend
fastapi dev inference_medsam2.py
```

The backend API will be available at `http://localhost:8000`

### Starting the Frontend Server

In a new terminal, navigate to the frontend directory and start the React development server:

```bash
cd frontend
npm run dev 
```

The frontend application will be available at `http://localhost:8080`

### Accessing the Application

Once both servers are running:

1. Open your web browser
2. Navigate to `http://localhost:8080`
3. Begin uploading surgical videos and creating annotations

## API Documentation

With the backend server running, you can access the interactive API documentation at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`


## Development Workflow

1. **Upload Video**: Select and upload surgical video files
2. **Initialize Segmentation**: Start adding points without labels for instance segementation, or add labels for semantic segmentation. More unique labels may result in OutOfMemory errors or long processing times. 
3. **Create Annotations**: Add positive or negative points on regions of interest, negative points are for refining the initial prompt that will be fed into the model.
4. **Assign Labels**: Provide descriptive labels for tracked elements
5. **Process Results**: Review and export segmentation results
6. **Collaborate**: Share results with team members for review

## Contributing

This is a proof-of-concept application under active development. Contributions are welcome through:

- Bug reports and feature requests
- Code contributions via pull requests
- Documentation improvements
- Testing and validation feedback

## Technology Stack

- **Frontend**: React, TypeScript, ShadCN 
- **Backend**: FastAPI, Python, Pydantic
- **AI/ML**: MedSAM2, PyTorch, OpenCV

## Roadmap

- [ ] Enhanced video processing capabilities
- [ ] Multi-user collaboration features
- [ ] Integration with medical imaging standards (DICOM)
- [ ] Mobile application support
- [ ] Real-time Inference

## Support

For technical support or questions regarding the MedSAM2 Interface:

- Create an issue in the GitHub repository
- Contact the UnivLabs development team
- Refer to the documentation in the `docs/` directory

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **UnivLabs** for the Collaborative Surgery Module initiative
- **MedSAM2** development team for the underlying AI technology
- **FastAPI** and **React** communities for excellent frameworks

---

**Note**: This is a proof-of-concept application. For production use, additional security measures, testing, and optimization should be implemented.# About:

